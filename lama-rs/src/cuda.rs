//! Runtime CUDA backend.
//!
//! The driver is reached through the `lightgpu` toolkit, which resolves it with
//! `dlopen` at run time, so the binary runs on hosts with no CUDA and simply
//! falls back to the CPU engine.  The device kernels are compiled by `build.rs`
//! into two fatbins and embedded below: `cuda/lama.cu` (this engine's own
//! kernels) and the toolkit subset this engine calls (the batched 2-D
//! transforms).  They are loaded as two `lightgpu::vm::Module`s, and every
//! kernel is launched by name through `lightgpu::vm::Args`; which module
//! defines a name is decided by its `k_` / `lg_` prefix.
//!
//! Each embedded image holds SASS for the compute capabilities in
//! `DEFAULT_ARCHES` (what a load failure on an unlisted GPU reports), plus PTX
//! for compute 8.0 so a device newer than that list still has something the
//! driver can compile.  The
//! driver ignores the PTX entry whenever a matching SASS one exists, and a PTX
//! entry only ever applies to a device at least as new as its `.target`, so the
//! SASS targets are what actually keep Pascal through Ampere free of JIT.
//!
//! Numerical contract: identical arithmetic to `cpu.rs`.  Convolutions are
//! im2col + the slab SGEMM (single precision, no implicit conversions), the
//! Fourier unit calls the shared toolkit's batched 2-D transforms
//! (`lg_fft2_r2c` / `lg_fft2_c2r`, compiled from lightgpu's kernels.cu into a
//! second module) with the same `1/sqrt(h*w)` ortho scaling, and the
//! half-spectrum completion reproduces the verified 2-D Hermitian reflection
//! rule from `irfft2_ortho`.

use std::ffi::c_void;
use std::ptr;

use lightgpu::vm::{Args, Launch, Module};

use crate::model::{ConvRef, FfcBlock, Model, Step};
use crate::weights::WeightStore;

/// The engine's own kernels (`cuda/lama.cu`): SASS for each `sm_*` target in
/// `DEFAULT_ARCHES`, plus PTX for compute 8.0 as the forward-compatibility path
/// (verifiably working - the JIT output on this machine is bit-identical to the
/// SASS).
const PROJECT_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/lama_project.fatbin"));

/// The toolkit kernels this engine calls (`lightgpu`'s `cuda/kernels.cu`): the
/// batched 2-D Fourier transforms behind the spectral blocks.  Compiled as a
/// separate module by `build.rs`, so a name here cannot shadow a name there.
const TOOLKIT_IMAGE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/lama_toolkit.fatbin"));

/// What a `CUresult` means here, in terms of the userspace/kernel driver split.
/// A bare code is not much use: the interesting failure is a `libcuda.so.1` that
/// is older than the loaded kernel module, which fails with
/// `CUDA_ERROR_FORWARD_COMPATIBILITY` (804).  The kernel module's own version
/// picks that apart; a driver that will not load at all is reported by
/// `lightgpu::ffi::driver` itself, which names `LA_CUDA_LIB` in its error.
fn context_note(code: i32) -> String {
    let mut lines = vec![format!("CUDA is not available in this process ({code}).")];
    if let Ok(text) = std::fs::read_to_string("/proc/driver/nvidia/version") {
        if let Some(v) = text.lines().next().and_then(|l| l.split_whitespace().nth(7)) {
            lines.push(format!("  nvidia kernel module: {v}"));
        }
    }
    if code == 804 {
        lines.push("  libcuda.so.1 does not match the loaded kernel module, so bringing".to_string());
        lines.push("  up a context fails with error 804.  Point LD_LIBRARY_PATH at the".to_string());
        lines.push("  userspace driver matching the module, e.g.".to_string());
        lines.push("    LD_LIBRARY_PATH=/path/to/your/nvidia/driver/lib".to_string());
    }
    lines.join("\n")
}

/// A device buffer of `len` f32.
struct Dev {
    ptr: *mut c_void,
    len: usize,
    /// False for windows into another buffer, which the owner frees.
    owned: bool,
}

impl Drop for Dev {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            // SAFETY: `ptr` came out of `DevBuf::alloc`, i.e. `cuMemAlloc`, and
            // this is the one place a `Dev` is given back.
            if !pool_release(self.ptr, self.len.max(1) * 4) {
                if let Some(c) = Cuda::get() {
                    unsafe { (c.driver.cuMemFree)(self.ptr as lightgpu::ffi::CUdeviceptr) };
                }
            }
        }
    }
}

/// A size-bucketed device memory pool.
/// `cuMemAlloc`/`cuMemFree` cost about 100 us per pair on this driver and each
/// `cuMemAlloc` implicitly synchronises the context, so a pass that makes a
/// thousand short-lived allocations loses a large fraction of its time to the
/// allocator alone.  Freed blocks are kept in a free list keyed by their exact
/// byte size (activations and patch matrices come in a handful of sizes and are
/// recycled in a tight loop, so the hit rate is near total).
/// The pool is deliberately simple: exact-size matching, a per-size list, and a
/// global byte cap.  Blocks are never handed back to the driver except when the
/// cap forces it, and the process exit reclaims whatever is left.
struct Pool {
    free: std::collections::HashMap<usize, Vec<*mut c_void>>,
    bytes: usize,
    hits: usize,
    misses: usize,
}

thread_local! {
    static POOL: std::cell::RefCell<Pool> = std::cell::RefCell::new(Pool {
        free: std::collections::HashMap::new(),
        bytes: 0,
        hits: 0,
        misses: 0,
    });
}

/// Keep at most this many bytes of freed device memory around.
const POOL_CAP: usize = 768 << 20;

fn pool_enabled() -> bool {
    // Checked once: the env var never changes mid-run.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("LAMA_NO_POOL").is_err())
}

/// A boolean environment flag, resolved once per process.
/// The hot paths used to call `std::env::var` (a getenv plus a heap allocation
/// and a string compare) on every convolution, every upsample and every
/// `op_scope`, i.e. tens of times per forward pass.  Env vars cannot change
/// mid-run, so each flag is cached in a `OnceLock` at first use.
fn flag(name: &'static str) -> bool {
    static FLAGS: std::sync::OnceLock<std::collections::HashMap<&'static str, bool>> =
        std::sync::OnceLock::new();
    let m = FLAGS.get_or_init(|| {
        let mut m = std::collections::HashMap::new();
        for n in [
            "LAMA_CONVT_GATHER",
            "LAMA_PROFILE_SUB",
            "LAMA_PROFILE_SYNC",
            "LAMA_FFT_HOST",
            "LAMA_PROFILE_OPS",
            "LAMA_PROFILE",
            "LAMA_NO_POOL",
        ] {
            m.insert(n, std::env::var(n).is_ok());
        }
        m
    });
    *m.get(name).unwrap_or(&false)
}

/// Take a block of exactly `bytes` from the pool, if one is free.
fn pool_acquire(bytes: usize) -> Option<*mut c_void> {
    if !pool_enabled() {
        return None;
    }
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        let list = p.free.get_mut(&bytes)?;
        let ptr = list.pop()?;
        if list.is_empty() {
            p.free.remove(&bytes);
        }
        p.bytes -= bytes;
        p.hits += 1;
        Some(ptr)
    })
}

/// Return a block to the pool.  Returns false if the caller must free it.
fn pool_release(ptr: *mut c_void, bytes: usize) -> bool {
    if !pool_enabled() {
        return false;
    }
    POOL.with(|p| {
        let mut p = p.borrow_mut();
        if p.bytes + bytes > POOL_CAP {
            return false;
        }
        p.free.entry(bytes).or_default().push(ptr);
        p.bytes += bytes;
        true
    })
}

/// The engine's device state: the two loaded modules and the weight blob.
/// Reaching the driver - and with it every device allocation, copy and launch -
/// is `lightgpu`'s job; what is left here is the part that belongs to this
/// engine: which module holds which kernel, and the one device copy of the
/// weights.  The handle is a `&'static` from the toolkit's own cache, so this
/// type holds no library reference and needs no `Send`/`Sync` proof of its own:
/// it is opened once, on the thread that opens it, and only ever read after.
// SAFETY: `Cuda` holds only process-wide handles - the toolkit's driver, two
// loaded modules and a device pointer - and the engine runs it from one thread;
// the `Rc` in `blob_arena` is never cloned across threads.  The same reasoning
// covers the `Lib` handles this type used to hold.
unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}

pub struct Cuda {
    driver: &'static lightgpu::ffi::Driver,
    /// The engine's own kernels (`lama_project.fatbin`).
    module: Module,
    /// The toolkit subset this engine calls (`lama_toolkit.fatbin`): the batched
    /// 2-D Fourier transforms.  A separate module, so a name here cannot shadow
    /// the engine's own kernels.
    toolkit: Module,
    /// One device block holding a verbatim copy of the whole weight blob.
    ///
    /// The blob is a single contiguous mmap of 989 tensors with no padding, so
    /// one copy reproduces every tensor at `arena + info.offset`.  This replaces
    /// ~989 individual uploads, each of which cost about 111 us of pure API
    /// latency, with one copy of 204 MB that runs at PCIe speed.
    blob_arena: std::sync::OnceLock<std::rc::Rc<Dev>>,
}

/// The outcome of the open attempt, kept separately from the `Cuda` it produces.
/// `Cuda::get` answers "is there a GPU here", and the only honest answer comes
/// from opening one.  A later failure - `device()`, say - must not turn that
/// into "no GPU", or the caller would silently take the CPU path, so the two are
/// cached apart and only the open result decides.
static CUDA: std::sync::OnceLock<Result<Cuda, String>> = std::sync::OnceLock::new();
static CUDA_RESULT: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();

impl Cuda {
    fn get() -> Option<&'static Cuda> {
        match CUDA.get_or_init(Cuda::open) {
            Ok(c) => Some(c),
            Err(_) => None,
        }
    }

    /// Bring up the device and load both kernel images.
    ///
    /// `lightgpu::vm::device` is what actually reaches the driver: it opens
    /// `libcuda.so.1`, initialises it and makes the primary context current (see
    /// `lightgpu::vm::bind_context`), so a failure there is the one failure the
    /// CPU fallback gets to explain.  Module loading follows, and by then a
    /// missing or unloadable image is a build problem rather than a missing GPU.
    fn open() -> Result<Cuda, String> {
        let t_all = std::time::Instant::now();
        let device = lightgpu::vm::device().map_err(|e| format!("{e}\n{}", context_note(0)))?;
        if flag("LAMA_PROFILE") {
            eprintln!(
                "gpu ctx: {} on {} (sm_{}{}, {} SMs)",
                device.driver.lib_path, device.name, device.cc_major, device.cc_minor, device.sm_count
            );
        }
        let r = (|| -> Result<Cuda, String> {
            let t = std::time::Instant::now();
            let module = Module::load(PROJECT_IMAGE)?;
            let toolkit = Module::load(TOOLKIT_IMAGE)?;
            if flag("LAMA_PROFILE") {
                eprintln!("gpu ctx: fatbins {:.3}s", t.elapsed().as_secs_f32());
            }
            Ok(Cuda {
                driver: device.driver,
                module,
                toolkit,
                blob_arena: std::sync::OnceLock::new(),
            })
        })();
        match &r {
            Ok(_) => {
                let _ = CUDA_RESULT.set(Ok(()));
            }
            Err(e) => {
                let _ = CUDA_RESULT.set(Err(e.clone()));
            }
        }
        if flag("LAMA_PROFILE") {
            eprintln!("gpu ctx total: {:.3}s", t_all.elapsed().as_secs_f32());
        }
        r
    }

    /// The module that defines `name`.
    ///
    /// The two images are loaded as two modules because a function handle is only
    /// valid for the module it came from, so a launch has to name the right one.
    /// The engine's kernels are all `k_*` and the toolkit's `lg_*`, which is how
    /// this picks without a table.
    fn module_of(&self, name: &str) -> Result<&Module, String> {
        // Counting here keeps the launch tally exact: every launch site calls
        // this exactly once, and nothing else does.
        LAUNCHES.with(|c| c.set(c.get() + 1));
        if name.starts_with("lg_") {
            return Ok(&self.toolkit);
        }
        Ok(&self.module)
    }

    /// Name a driver failure: `cuGetErrorName` already knows every code.
    fn err(&self, code: lightgpu::ffi::CUresult, what: &str) -> String {
        format!("{what}: {} ({code})", self.driver.error_name(code))
    }

    /// Allocate through the driver, at the front of the pool if it can help.
    ///
    /// This has to try the pool first: `cuMemAlloc` implicitly synchronises the
    /// context, and a pass makes hundreds of short-lived allocations.
    fn alloc(&self, len: usize) -> Result<Dev, String> {
        let bytes = len.max(1) * 4;
        if let Some(p) = pool_acquire(bytes) {
            return Ok(Dev { ptr: p, len, owned: true });
        }
        let t = std::time::Instant::now();
        let mut p: lightgpu::ffi::CUdeviceptr = 0;
        // SAFETY: `p` is a valid out-parameter.
        let r = unsafe { (self.driver.cuMemAlloc)(&mut p, bytes) };
        if r != lightgpu::ffi::CUDA_SUCCESS {
            return Err(self.err(r, "cuMemAlloc"));
        }
        let p = p as *mut c_void;
        ALLOC_STATS.with(|s| {
            let mut s = s.borrow_mut();
            s.0 += 1;
            s.1 += t.elapsed().as_secs_f32();
        });
        POOL.with(|s| s.borrow_mut().misses += 1);
        Ok(Dev { ptr: p, len, owned: true })
    }

    /// Upload the entire weight blob in ONE `cuMemcpyHtoD`.
    ///
    /// The blob's tensors are contiguous from offset 0 with no padding, so the
    /// device copy is a faithful image of the host mapping and every tensor can
    /// be addressed as `arena + offset` - see `Slice`.  Doing this once costs
    /// about as much as a fifth of the individual uploads it replaces.
    fn blob_arena(&self, blob: &[u8]) -> Result<std::rc::Rc<Dev>, String> {
        // Built on first use and reused for the process: the blob never changes.
        if let Some(a) = self.blob_arena.get() {
            return Ok(a.clone());
        }
        let t_arena = std::time::Instant::now();
        let n = (blob.len() + 3) / 4;
        // Straight from the driver, never through the pool: the pool's whole
        // purpose is to recycle short-lived activations, and this block lives
        // for the process.  If it came from the pool it would be released back
        // when the `Rc` drops and handed to the next activation, which would
        // overwrite the weights.
        let mut p: lightgpu::ffi::CUdeviceptr = 0;
        // SAFETY: `p` is a valid out-parameter.
        let r = unsafe { (self.driver.cuMemAlloc)(&mut p, n.max(1) * 4) };
        if r != lightgpu::ffi::CUDA_SUCCESS {
            return Err(self.err(r, "cuMemAlloc (weight arena)"));
        }
        let p = p as *mut c_void;
        let d = Dev { ptr: p, len: n, owned: true };
        // SAFETY: source is `blob.len()` bytes, destination is that many rounded
        // up to a whole number of floats.
        let r = unsafe {
            (self.driver.cuMemcpyHtoD)(d.ptr as lightgpu::ffi::CUdeviceptr, blob.as_ptr() as *const c_void, blob.len())
        };
        if r != lightgpu::ffi::CUDA_SUCCESS {
            return Err(self.err(r, "cuMemcpyHtoD (weight arena)"));
        }
        let rc = std::rc::Rc::new(d);
        let _ = self.blob_arena.set(rc.clone());
        if flag("LAMA_PROFILE") {
            eprintln!(
                "gpu weight arena: {:.1} MB in {:.3}s ({:.1} GB/s)",
                blob.len() as f32 / 1e6,
                t_arena.elapsed().as_secs_f32(),
                blob.len() as f32 / 1e9 / t_arena.elapsed().as_secs_f32().max(1e-6)
            );
        }
        Ok(rc)
    }

    /// A non-owning `Dev` window into the blob arena for a verbatim tensor.
    ///
    /// The arena is cached in `blob_arena` and, being a `OnceLock` owned by the
    /// process-wide `Cuda`, is never dropped, so the pointer stays valid for the
    /// life of the program and the `Dev` can safely be `owned: false`.
    fn blob_view(&self, store: &WeightStore, name: &str) -> Result<Dev, String> {
        let info = store.info(name)?.clone();
        let arena = self.blob_arena(store.bytes())?;
        // SAFETY: `info.offset + info.nbytes` was validated against the blob
        // length by `WeightStore::validate`, and the arena is a copy of the
        // whole blob, so the window is inside the allocation.
        Ok(Dev {
            ptr: unsafe { (arena.ptr as *mut u8).add(info.offset) as *mut c_void },
            len: info.nbytes / 4,
            owned: false,
        })
    }

    /// One host-to-device copy, through the driver.
    fn upload(&self, data: &[f32]) -> Result<Dev, String> {
        let d = self.alloc(data.len())?;
        if !data.is_empty() {
            // SAFETY: source is a valid slice, destination has len*4 bytes.
            let bytes = data.len() * 4;
            let r = unsafe {
                (self.driver.cuMemcpyHtoD)(d.ptr as lightgpu::ffi::CUdeviceptr, data.as_ptr() as *const c_void, bytes)
            };
            if r != lightgpu::ffi::CUDA_SUCCESS {
                return Err(self.err(r, "cuMemcpyHtoD"));
            }
        }
        Ok(d)
    }

    /// One device-to-host copy, through the driver.
    fn download(&self, d: &Dev) -> Result<Vec<f32>, String> {
        // Synchronise first: `cuMemcpyDtoH` is asynchronous in the sense that it
        // returns once the copy is queued, and the host buffer is not read back
        // through any stream-ordered mechanism.
        self.sync()?;
        let mut out = vec![0f32; d.len];
        if d.len > 0 {
            // SAFETY: destination is a valid slice, source has len*4 bytes.
            let bytes = d.len * 4;
            let r = unsafe {
                (self.driver.cuMemcpyDtoH)(out.as_mut_ptr() as *mut c_void, d.ptr as lightgpu::ffi::CUdeviceptr, bytes)
            };
            if r != lightgpu::ffi::CUDA_SUCCESS {
                return Err(self.err(r, "cuMemcpyDtoH"));
            }
        }
        Ok(out)
    }

    /// Run one kernel through the toolkit.
    ///
    /// Every call site uses a single-dimensional grid and block on the default
    /// stream, which is the same ordering the engine had when it created a stream
    /// of its own: the host is single-threaded here, and every copy in this file
    /// either synchronises or is followed by one.
    fn launch_at(
        &self,
        aa: &mut Args,
        name: &str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<(), String> {
        aa.launch(self.module_of(name)?, name, Launch::new(grid, block))
    }

    /// Wait for everything queued so far.
    fn sync(&self) -> Result<(), String> {
        lightgpu::vm::sync()
    }
}

// Count of kernels launched, for attributing per-op launch overhead.
thread_local! {
    static LAUNCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn grid_for(n: usize, block: u32) -> u32 {
    ((n as u64 + block as u64 - 1) / block as u64) as u32
}

const BLOCK: u32 = 256;

// ---------------------------------------------------------------------------
// Device activations and weights
// ---------------------------------------------------------------------------

/// A device-resident activation block `[c][h][w]`, NCHW row-major.
struct DActs {
    buf: Dev,
    c: usize,
    h: usize,
    w: usize,
}

impl DActs {
    fn new(c: usize, h: usize, w: usize) -> Result<Self, String> {
        let ctx = ctx()?;
        Ok(DActs { buf: ctx.alloc(c * h * w)?, c, h, w })
    }

    fn plane(&self) -> usize {
        self.h * self.w
    }

    /// A non-owning view of `channels` planes starting at `first`.
    ///
    /// The returned value borrows the parent buffer; it must not outlive it.
    /// Its `Dev` is marked `owned: false` so `Drop` leaves the pointer alone.
    fn view(&self, first: usize, channels: usize) -> DActs {
        DActs {
            buf: Dev {
                ptr: unsafe { (self.buf.ptr as *mut u8).add(first * self.plane() * 4) as *mut c_void },
                len: channels * self.plane(),
                owned: false,
            },
            c: channels,
            h: self.h,
            w: self.w,
        }
    }
}

/// A convolution weight `[cout][cin][kh][kw]` plus optional bias, on device.
struct DConv {
    w: Dev,
    bias: Option<Dev>,
    cin: usize,
    cout: usize,
    kh: usize,
    kw: usize,
    /// The same weights reordered to `[cin][cout][kh][kw]`, used by the direct
    /// 7x7 kernel which walks one input channel at a time.  Only built for the
    /// shapes that kernel accepts.
    w_t: Option<Dev>,
}

/// Folded BatchNorm on device.
struct DBn {
    scale: Dev,
    shift: Dev,
    channels: usize,
}

#[allow(dead_code)]  // Mirrors `FfcWeights`: geometry carried for clarity.
struct DFfc {
    kernel: usize,
    stride: usize,
    pad: usize,
    reflect: bool,
    in_local: usize,
    in_global: usize,
    out_local: usize,
    out_global: usize,
    l2l: Option<DConv>,
    l2g: Option<DConv>,
    g2l: Option<DConv>,
    bn_l: Option<DBn>,
    bn_g: Option<DBn>,
    spectral: Option<DSpectral>,
}

/// The spectral (global-to-global) branch, mirroring `cpu::Spectral`:
///   pooled = avg_pool2x2(x) when stride == 2 else x
///   feat   = ReLU(bn1(conv1(pooled)))                     1x1, half -> half
///   fu     = irfft(bn_fu(fu_conv(stack(rfft(feat)))))     1x1 over 2*half
///   out    = conv2(feat + fu)                             1x1, half -> half
/// The FourierUnit adds the pre-transform `feat` to its inverse transform, and
/// the real/imaginary stacks are the channels its 1x1 convolution sees.
struct DSpectral {
    conv1: DConv,
    bn1: DBn,
    fu_conv: DConv,
    fu_bn: DBn,
    conv2: DConv,
    half: usize,
    stride: usize,
}

struct DConvT {
    w: Dev,
    bias: Dev,
    cin: usize,
    cout: usize,
    /// Per-phase weight matrices `[cout][cin*taps]` row-major for the phase
    /// GEMM path, in phase order `py*2 + px` with each phase's taps ordered as
    /// `kernels.cu`'s `CONVT_KY`/`CONVT_KX` tables list them.  `None` falls back
    /// to the gather kernel.
    phases: Option<Vec<Dev>>,
}

/// Tap sets per output phase for stride 2, pad 1, k 3; must match
/// `CONVT_KY`/`CONVT_KX` in kernels.cu.
const CONVT_TAPS: [[(usize, usize); 4]; 4] = [
    [(1, 1), (0, 0), (0, 0), (0, 0)],
    [(1, 0), (1, 2), (0, 0), (0, 0)],
    [(0, 1), (2, 1), (0, 0), (0, 0)],
    [(0, 0), (0, 2), (2, 0), (2, 2)],
];

/// Number of contributing taps per phase, matching `CONVT_TAPS` above and the
/// `CONVT_KY`/`CONVT_KX` tables in kernels.cu.
const CONVT_TAP_COUNT: [usize; 4] = [1, 2, 2, 4];

enum DStep {
    ReflectPad(usize),
    Ffc(DFfc),
    Res(DFfc, DFfc),
    Concat,
    Upsample(DConvT, DBn),
    OutConv(DConv),
}

fn ctx() -> Result<&'static Cuda, String> {
    Cuda::get().ok_or_else(|| match CUDA.get() {
        Some(Err(e)) => e.clone(),
        _ => "CUDA is unavailable".to_string(),
    })
}

/// Load-time staging for *derived* device tensors.
/// Most of the model is verbatim in the blob and now costs nothing to upload
/// (see `Cuda::blob_view`).  What is left are tensors the loader *computes* -
/// the folded BatchNorm scale/shift, the 7x7 weight reorder and the transposed
/// convolution's phase matrices - and there are 152 BatchNorms, so those tiny
/// uploads alone were 304 `cuMemcpyHtoD` calls of a few hundred bytes each, every
/// one of them paying the full per-call latency.
/// `Staging` packs all of them into one host buffer; `finish` uploads that
/// buffer with a single `cuMemcpyHtoD` and returns the arena.  Records are
/// `(byte_offset_in_arena, len_in_floats)`, and `window` turns one back into a
/// non-owning `Dev` pointing into the arena.
#[derive(Default)]
struct Staging {
    host: Vec<f32>,
    records: Vec<(usize, usize)>,
}

thread_local! {
    /// The load-time staging buffer, shared by every `load_*` helper.
    ///
    /// Threading it through the loader signatures would touch every function for
    /// no benefit: the whole load runs on one thread, once per process.
    static STAGING: std::cell::RefCell<Staging> = std::cell::RefCell::new(Staging::default());
    /// The uploaded staging arena's base pointer.
    ///
    /// A bare pointer rather than an `Rc<Dev>`, deliberately: these arenas live
    /// for the whole process, and a `Dev` stored in a thread-local would be
    /// dropped by the TLS destructor at exit - at which point `pool_release`
    /// would reach for the allocator's own thread-locals, which may already be
    /// gone, and the panic inside the destructor aborts the process *after* the
    /// result has been written.  The memory is intentionally never freed.
    static STAGING_ARENA: std::cell::Cell<*mut c_void> =
        const { std::cell::Cell::new(ptr::null_mut()) };
}

/// Record a derived tensor's values for the single staging upload, returning the
/// record index that `staged` later turns into a device window.
fn stage(data: &[f32]) -> usize {
    STAGING.with(|s| s.borrow_mut().push(data))
}

impl Staging {
    /// Copy `data.len()` floats into the buffer and return the record index.
    fn push(&mut self, data: &[f32]) -> usize {
        let off = self.host.len() * 4;
        self.host.extend_from_slice(data);
        // Round the next tensor up to a 16-byte boundary so every window is
        // aligned for vectorised kernel access.
        while self.host.len() * 4 % 16 != 0 {
            self.host.push(0.0);
        }
        self.records.push((off, data.len()));
        self.records.len() - 1
    }
}

/// Upload every staged tensor in ONE `cuMemcpyHtoD` and remember the arena.
/// Called once, as soon as every folded BatchNorm has been recorded.
fn stage_commit() -> Result<(), String> {
    let host = STAGING.with(|s| std::mem::take(&mut s.borrow_mut().host));
    let c = ctx()?;
    // Straight from the driver, never the pool: this block lives for the process
    // and must never be recycled as an activation.  It is also never freed, so
    // no thread-local destructor has to touch the allocator at exit.
    let n = host.len();
    let mut p: lightgpu::ffi::CUdeviceptr = 0;
    // SAFETY: `p` is a valid out-parameter.
    let r = unsafe { (c.driver.cuMemAlloc)(&mut p, n.max(1) * 4) };
    if r != lightgpu::ffi::CUDA_SUCCESS {
        return Err(c.err(r, "cuMemAlloc (staged weights)"));
    }
    if n > 0 {
        // SAFETY: source has `n` floats, destination `n*4` bytes.
        let r = unsafe { (c.driver.cuMemcpyHtoD)(p as lightgpu::ffi::CUdeviceptr, host.as_ptr() as *const c_void, n * 4) };
        if r != lightgpu::ffi::CUDA_SUCCESS {
            return Err(c.err(r, "cuMemcpyHtoD (staged weights)"));
        }
    }
    STAGING_ARENA.with(|a| a.set(p as *mut c_void));
    Ok(())
}

/// A device window for a staged tensor, valid after `stage_commit`.
fn staged(i: usize) -> Dev {
    let (off, len) = STAGING.with(|s| s.borrow().records[i]);
    STAGING_ARENA.with(|a| {
        let p = a.get();
        if p.is_null() {
            panic!("staged tensor used before stage_commit");
        }
        Dev {
            // SAFETY: `off + len*4` lies inside the arena, which is a copy of
            // exactly the host buffer the record was taken from.
            ptr: unsafe { (p as *mut u8).add(off) as *mut c_void },
            len,
            owned: false,
        }
    })
}

fn load_conv(store: &WeightStore, prefix: &str) -> Result<DConv, String> {
    let info = store.info(&format!("{prefix}.weight"))?.clone();
    // shapes are [cout][cin][kh][kw]
    let (cout, cin, kh, kw) = (info.shape[0], info.shape[1], info.shape[2], info.shape[3]);
    let c = ctx()?;
    // The weight tensor is byte-identical in the blob and on the device, so it
    // is addressed inside the single arena copy rather than uploaded again.
    let w_dev = c.blob_view(store, &format!("{prefix}.weight"))?;
    let bias_dev = match store.info(&format!("{prefix}.bias")) {
        Ok(bi) if bi.numel() == cout => Some(c.blob_view(store, &format!("{prefix}.bias"))?),
        _ => None,
    };
    // The direct 7x7 kernel loops input channels outermost, so it wants
    // [cin][cout][kh][kw]; build that once here rather than per call.
    let w_t = if kh == 7 && kw == 7 && cout <= 8 {
        let w = store.f32(&format!("{prefix}.weight"))?;
        let mut t = vec![0f32; w.len()];
        for co in 0..cout {
            for ci in 0..cin {
                for ky in 0..kh {
                    for kx in 0..kw {
                        t[((ci * cout + co) * kh + ky) * kw + kx] = w[((co * cin + ci) * kh + ky) * kw + kx];
                    }
                }
            }
        }
        Some(c.upload(&t)?)
    } else {
        None
    };
    Ok(DConv {
        w: w_dev,
        bias: bias_dev,
        cin,
        cout,
        kh,
        kw,
        w_t,
    })
}

/// Fold one BatchNorm's four tensors into `(scale, shift)`, as `cpu::Bn::load`.
fn fold_bn(store: &WeightStore, prefix: &str) -> Result<(Vec<f32>, Vec<f32>), String> {
    let w = store.f32(&format!("{prefix}.weight"))?;
    let b = store.f32(&format!("{prefix}.bias"))?;
    let mean = store.f32(&format!("{prefix}.running_mean"))?;
    let var = store.f32(&format!("{prefix}.running_var"))?;
    let mut scale = vec![0f32; w.len()];
    let mut shift = vec![0f32; w.len()];
    for i in 0..w.len() {
        let s = w[i] / (var[i] + 1e-5).sqrt();
        scale[i] = s;
        shift[i] = b[i] - mean[i] * s;
    }
    Ok((scale, shift))
}

/// Fold and upload every BatchNorm in the model, once.
/// Every BatchNorm is identified by its `running_var` tensor, and there are 152
/// of them, i.e. 304 folded tensors - a few hundred bytes each, but 304 separate
/// `cuMemcpyHtoD` calls at roughly 100 us of latency apiece.  They are packed into
/// one buffer (~317 KB in total, since the whole model has 40,576 BN channels)
/// and uploaded with a single copy instead.
fn stage_bns(store: &WeightStore) -> Result<(), String> {
    let mut names: Vec<String> = store
        .index
        .order
        .iter()
        .filter(|n| n.ends_with(".running_var"))
        .map(|n| n.trim_end_matches(".running_var").to_string())
        .collect();
    // Stable order so the record indices are reproducible, though the lookup
    // table below is keyed by name and does not depend on it.
    names.sort();
    let mut index = std::collections::HashMap::with_capacity(names.len() * 2);
    for name in &names {
        let (scale, shift) = fold_bn(store, name)?;
        // `stage` records in the first pass and only advances the cursor in the
        // second, so the two passes stay locked together - the index table is
        // kept from the recording pass.
        let si = stage(&scale);
        let ti = stage(&shift);
        index.insert(format!("{name}.scale"), si);
        index.insert(format!("{name}.shift"), ti);
    }
    stage_commit()?;
    BN_INDEX.with(|i| *i.borrow_mut() = Some(index));
    Ok(())
}

// Record indices of every staged BatchNorm tensor, keyed
// `"<module>.scale"` / `"<module>.shift"`, filled by `stage_bns`.
thread_local! {
    static BN_INDEX: std::cell::RefCell<Option<std::collections::HashMap<String, usize>>> =
        const { std::cell::RefCell::new(None) };
}

/// BatchNorm is *derived*, not verbatim: the two device tensors are
/// `w/sqrt(var+eps)` and `b - mean*scale`, which do not appear in the blob.  They
/// are staged together with every other derived tensor, so each `DBn` holds
/// windows into one shared arena rather than owning two tiny allocations.
fn load_bn(store: &WeightStore, prefix: &str, channels: usize) -> Result<DBn, String> {
    // The folded values themselves are produced by `stage_bns`, which walks
    // every BatchNorm in one go; this call only needs the module's recorded
    // indices.  `fold_bn` here would be a duplicate fold whose result was
    // discarded, so the channel count is checked from the record instead.
    if BN_INDEX.with(|i| i.borrow().is_none()) {
        // The first BatchNorm folds the whole model's BNs at once; the upload
        // itself is committed after the entire traversal (see `load_steps`).
        stage_bns(store)?;
    }
    // The two record indices are looked up by module name, so no ordering
    // assumption is needed between this call and `stage_bns`'s traversal.
    let si = BN_INDEX.with(|i| {
        i.borrow()
            .as_ref()
            .and_then(|m| m.get(&format!("{prefix}.scale")).copied())
    });
    let ti = BN_INDEX.with(|i| {
        i.borrow()
            .as_ref()
            .and_then(|m| m.get(&format!("{prefix}.shift")).copied())
    });
    let (si, ti) = match (si, ti) {
        (Some(a), Some(b)) => (a, b),
        _ => return Err(format!("BatchNorm {prefix} was not staged")),
    };
    let dbn = DBn { scale: staged(si), shift: staged(ti), channels };
    if dbn.scale.len != channels || dbn.shift.len != channels {
        return Err(format!(
            "BatchNorm {prefix}: staged {} scale / {} shift values, expected {channels}",
            dbn.scale.len, dbn.shift.len
        ));
    }
    Ok(dbn)
}

fn load_dffc(store: &WeightStore, b: &FfcBlock) -> Result<DFfc, String> {
    let prefix = format!("model.{}", b.index);
    let ffc_prefix = match b.sub {
        0 => format!("{prefix}.ffc"),
        1 => format!("{prefix}.conv1.ffc"),
        _ => format!("{prefix}.conv2.ffc"),
    };
    let partial = match b.sub {
        0 => prefix.clone(),
        1 => format!("{prefix}.conv1"),
        _ => format!("{prefix}.conv2"),
    };
    let conv = |r: &ConvRef| -> Result<Option<DConv>, String> {
        match r {
            ConvRef::Absent => Ok(None),
            ConvRef::Present { name, .. } => Ok(Some(load_conv(store, &format!("{ffc_prefix}.{name}"))?)),
        }
    };
    let spectral = if b.g2g {
        // `Spectral::load` in cpu.rs uses the `convg2g` prefix: conv1/bn1 are
        // `conv1.0` and `conv1.1`, the FourierUnit is `fu.conv_layer`/`fu.bn`,
        // and conv2 closes the branch.
        let p = format!("{ffc_prefix}.convg2g");
        let half = b.out_global / 2;
        Some(DSpectral {
            conv1: load_conv(store, &format!("{p}.conv1.0"))?,
            bn1: load_bn(store, &format!("{p}.conv1.1"), half)?,
            fu_conv: load_conv(store, &format!("{p}.fu.conv_layer"))?,
            fu_bn: load_bn(store, &format!("{p}.fu.bn"), 2 * half)?,
            conv2: load_conv(store, &format!("{p}.conv2"))?,
            half,
            stride: b.stride,
        })
    } else {
        None
    };
    Ok(DFfc {
        kernel: b.kernel,
        stride: b.stride,
        pad: b.pad,
        reflect: b.reflection,
        in_local: b.in_local,
        in_global: b.in_global,
        out_local: b.out_local,
        out_global: b.out_global,
        l2l: conv(&b.l2l)?,
        l2g: conv(&b.l2g)?,
        g2l: conv(&b.g2l)?,
        bn_l: if b.bn_local {
            Some(load_bn(store, &format!("{partial}.bn_l"), b.out_local)?)
        } else {
            None
        },
        bn_g: if b.bn_global {
            Some(load_bn(store, &format!("{partial}.bn_g"), b.out_global)?)
        } else {
            None
        },
        spectral,
    })
}

/// Build the device step tree.
/// Only the folded BatchNorm tensors are staged (see `stage_bns`): they are the
/// overwhelming majority of the derived uploads - 152 modules, 304 tensors - and
/// staging them turns 304 `cuMemcpyHtoD` calls into one.  The transposed
/// convolution's twelve phase matrices stay individual uploads, because staging
/// them would require a two-pass traversal whose second pass re-walks the whole
/// model and rebuilds the step tree, and measured against the twelve uploads it
/// replaces that trade is a net loss (load_steps 0.072 s single-pass versus
/// 0.079-0.082 s two-pass).
fn load_steps(store: &WeightStore, model: &Model) -> Result<Vec<DStep>, String> {
    let mut steps = Vec::new();
    for step in &model.steps {
        match step {
            Step::ReflectPad(p) => steps.push(DStep::ReflectPad(*p)),
            Step::Ffc(b) => steps.push(DStep::Ffc(load_dffc(store, b)?)),
            Step::ResBlock(rb) => steps.push(DStep::Res(
                load_dffc(store, &rb.conv1)?,
                load_dffc(store, &rb.conv2)?,
            )),
            Step::Concat => steps.push(DStep::Concat),
            Step::Upsample(u) => {
                let prefix = format!("model.{}", u.index);
                let info = store.info(&format!("{prefix}.weight"))?.clone();
                let w = store.f32(&format!("{prefix}.weight"))?;
                // The bias is not read here: it is passed to the phase kernel as
                // a separate argument, straight from `blob_view`.
                let c = ctx()?;
                // Reorder the [cin][cout][3][3] weight into one matrix per output
                // phase, shaped `[cout][cin*taps]`, so the phase GEMM can consume
                // it directly.  `CONVT_TAPS` fixes the tap order and must match
                // `CONVT_KY`/`CONVT_KX` in kernels.cu.
                let (cin, cout) = (info.shape[0], info.shape[1]);
                // One matrix per phase, `[cout][cin*taps]` row-major, where the
                // column index is `ci*taps + ti` to match the gather kernel's
                // `col[(ci*taps + ti)*n + j]`.  Source layout is [cin][cout][3][3].
                let mut phases = Vec::with_capacity(4);
                for (ph, taps) in CONVT_TAPS.iter().enumerate() {
                    let taps_n = CONVT_TAP_COUNT[ph];
                    let k = cin * taps_n;
                    let mut m = vec![0f32; cout * k];
                    for ci in 0..cin {
                        for ti in 0..taps_n {
                            let (ky, kx) = taps[ti];
                            for co in 0..cout {
                                let src = ((ci * cout + co) * 3 + ky) * 3 + kx;
                                m[co * k + ci * taps_n + ti] = w[src];
                            }
                        }
                    }
                    phases.push(c.upload(&m)?);
                }
                let convt = DConvT {
                    w: c.blob_view(store, &format!("{prefix}.weight"))?,
                    bias: c.blob_view(store, &format!("{prefix}.bias"))?,
                    cin,
                    cout,
                    phases: Some(phases),
                };
                // The upsample BN is a bare `model.N` module (not `model.N.bn`).
                let bn = load_bn(store, &format!("model.{}", u.bn_index), u.cout)?;
                steps.push(DStep::Upsample(convt, bn));
            }
            Step::OutConv(o) => {
                steps.push(DStep::OutConv(load_conv(store, &format!("model.{}", o.index))?))
            }
        }
    }
    Ok(steps)
}

// ---------------------------------------------------------------------------
// Device operations
// ---------------------------------------------------------------------------

/// im2col + SGEMM for a kxk convolution.
/// The patch matrix is `[cin*kh*kw][oh*ow]` row-major with `lda = oh*ow`, and the
/// weights are `[cout][cin*kh*kw]` row-major with `ldb = cin*kh*kw`, so the single
/// `C[o][j] = sum_k B[o][k]*A[k][j]` SGEMM produces the `[cout][oh*ow]` row-major
/// output plane with no transpose work at either operand.
fn conv_forward(cv: &DConv, src: &DActs, pad: usize, stride: usize, reflect: bool) -> Result<DActs, String> {
    let c = ctx()?;
    if cv.kh == 1 && cv.kw == 1 && stride == 1 {
        return conv_1x1(cv, src);
    }
    // The 7x7 OutConv is the one shape where im2col dominates the arithmetic;
    // the direct kernel walks the receptive field instead.
    if cv.kh == 7 && cv.kw == 7 && stride == 1 && pad == 0 && cv.w_t.is_some() {
        return conv_7x7_direct(cv, src);
    }
    let oh = (src.h + 2 * pad - cv.kh) / stride + 1;
    let ow = (src.w + 2 * pad - cv.kw) / stride + 1;
    let k = cv.cin * cv.kh * cv.kw;
    let n = oh * ow;
    let col = c.alloc(k * n)?;
    let out = c.alloc(cv.cout * n)?;

    let t_im2col = std::time::Instant::now();
    let mut aa = Args::new();
    aa.ptr(src.buf.ptr as u64);
    aa.ptr(col.ptr as u64);
    aa.i32(cv.cin as i32);
    aa.i32(src.h as i32);
    aa.i32(src.w as i32);
    aa.i32(cv.kh as i32);
    aa.i32(cv.kw as i32);
    aa.i32(pad as i32);
    aa.i32(pad as i32);
    aa.i32(stride as i32);
    aa.i32(oh as i32);
    aa.i32(ow as i32);
    aa.i32(if reflect { 1i32 } else { 0i32 } as i32);
    c.launch_at(&mut aa, "k_im2col", (grid_for(k * n, BLOCK), 1, 1), (BLOCK, 1, 1))?;
    if flag("LAMA_PROFILE_SUB") {
        c.sync()?;
        SUB_STATS.with(|s| {
            let mut m = s.borrow_mut();
            let e = m.entry("im2col").or_insert((0usize, 0.0f32, 0usize, 0usize));
            e.0 += 1;
            e.1 += t_im2col.elapsed().as_secs_f32();
            // `cout` is the output channel count: the GEMM's M dimension is n,
            // its N dimension is cout and its K dimension is k.
            e.2 += n * cv.cout;
            e.3 += n * cv.cout * k;
        });
    }
    let t_gemm = std::time::Instant::now();
    sgemm(c, &col, &cv.w, &out, k, cv.cout, n)?;
    if flag("LAMA_PROFILE_SUB") {
        c.sync()?;
        SUB_STATS.with(|s| {
            let mut m = s.borrow_mut();
            let e = m.entry("sgemm").or_insert((0usize, 0.0f32, 0usize, 0usize));
            let t = t_gemm.elapsed().as_secs_f32();
            e.0 += 1;
            e.1 += t;
            e.2 += n * cv.cout;
            e.3 += 2 * n * cv.cout * k;
        });
    }

    if let Some(bias) = &cv.bias {
        bias_plane(c, &out, bias, cv.cout, n)?;
    }

    Ok(DActs { buf: out, c: cv.cout, h: oh, w: ow })
}

/// Direct 7x7 convolution without im2col, one thread per output pixel.
fn conv_7x7_direct(cv: &DConv, src: &DActs) -> Result<DActs, String> {
    let c = ctx()?;
    let w_t = cv.w_t.as_ref().ok_or("conv_7x7_direct without a transposed weight")?;
    let (oh, ow) = (src.h - 6, src.w - 6);
    let out = c.alloc(cv.cout * oh * ow)?;
    let bp = match &cv.bias {
        Some(b) => b.ptr,
        None => ptr::null_mut(),
    };
    // The padded input keeps its own extent: 518x518 in, 512x512 out.
    let mut aa = Args::new();
    aa.ptr(src.buf.ptr as u64);
    aa.ptr(w_t.ptr as u64);
    aa.ptr(bp as u64);
    aa.ptr(out.ptr as u64);
    aa.i32(cv.cin as i32);
    aa.i32(cv.cout as i32);
    aa.i32(src.h as i32);
    aa.i32(src.w as i32);
    aa.i32(oh as i32);
    aa.i32(ow as i32);
    c.launch_at(&mut aa, "k_conv7x7", (grid_for(oh * ow, BLOCK), 1, 1), (BLOCK, 1, 1))?;
    Ok(DActs { buf: out, c: cv.cout, h: oh, w: ow })
}

/// Add a per-channel bias to a `[cout][n]` plane in place, on the device.
fn bias_plane(c: &Cuda, out: &Dev, bias: &Dev, cout: usize, n: usize) -> Result<(), String> {
    let mut aa = Args::new();
    aa.ptr(out.ptr as u64);
    aa.ptr(bias.ptr as u64);
    aa.i32(cout as i32);
    aa.i64(n as i64);
    c.launch_at(&mut aa, "k_bias_plane", (grid_for(cout * n, BLOCK), 1, 1), (BLOCK, 1, 1))

}

/// The 1x1 case is a pure channel matmul over `[cin][n]` planes, so it runs as a
/// single SGEMM with no im2col at all.
// SGEMM: `c[cout][n] = b[cout][k] * a[k][n]`.  `k` must be a multiple of 4 and at
// least 4, and every shape the
// model's convolutions produce satisfies that (k is 9*cin, 4*cin or cin*taps, and
// cin is always a multiple of 64).  The OutConv is the exception (cout = 3) and
// never reaches here: it runs on k_conv7x7.  The kernel predicates its off-tile
// work itself, so the grid needs no exact-divisibility assumptions.
//
// The weight operand is addressed inside the engine's packed blob arena and is
// often not 16-byte aligned, so `b_align` is derived here and passed down.  The
// slab kernel stages B one scalar per thread and therefore does not consult it;
// it stays in the argument list because the kernel signature still takes it.
fn sgemm(
    c: &Cuda,
    a: &Dev,
    b: &Dev,
    out: &Dev,
    k: usize,
    cout: usize,
    n: usize,
) -> Result<(), String> {
    if k < 4 || k % 4 != 0 {
        return Err(format!(
            "sgemm: k must be a positive multiple of 4, got {k}"
        ));
    }
    // One 64-column tile per block, no shape dispatch: the slab kernel beats every
    // other configuration tried at every shape the engine produces.
    let nb_cols = grid_for(n, 64);
    // The weight tensors live at unpadded offsets inside the blob arena, so B is
    // frequently not 16-byte aligned (185 of the 223 GEMM weight tensors here sit
    // at 8 mod 16) and the kernel must not use a float4 load on it.  The kernel
    // picks its load width from this flag: 2 = float4, 1 = two float2s, 0 = four
    // scalars.
    let b_align_i = if (b.ptr as usize) % 16 == 0 {
        2i32
    } else if (b.ptr as usize) % 8 == 0 {
        1i32
    } else {
        0i32
    };
    
    let grid = nb_cols * grid_for(cout, 64);
    // `b_align` is not consulted by the slab kernel (its staging reads B one scalar at
    // a time, so any arena offset is fine), but it stays in the signature and argument
    // list; the value is still computed so the argument keeps its meaning.
    let mut aa = Args::new();
    aa.ptr(a.ptr as u64);
    aa.ptr(b.ptr as u64);
    aa.ptr(out.ptr as u64);
    aa.i32(n as i32);
    aa.i32(k as i32);
    aa.i32(n as i32);
    aa.i32(k as i32);
    aa.i32(cout as i32);
    aa.i32(n as i32);
    aa.i32(nb_cols as i32);
    aa.i32(b_align_i as i32);
    c.launch_at(&mut aa, "k_sgemm_slab", (grid, 1, 1), (BLOCK, 1, 1))

}

fn conv_1x1(cv: &DConv, src: &DActs) -> Result<DActs, String> {
    let c = ctx()?;
    let n = src.h * src.w;
    let out = c.alloc(cv.cout * n)?;
    let t_gemm = std::time::Instant::now();
    sgemm(c, &src.buf, &cv.w, &out, cv.cin, cv.cout, n)?;
    if flag("LAMA_PROFILE_SUB") {
        c.sync()?;
        SUB_STATS.with(|s| {
            let mut m = s.borrow_mut();
            let e = m.entry("sgemm1x1").or_insert((0usize, 0.0f32, 0usize, 0usize));
            e.0 += 1;
            e.1 += t_gemm.elapsed().as_secs_f32();
            e.2 += n * cv.cout;
            e.3 += 2 * n * cv.cout * cv.cin;
        });
    }
    if let Some(bias) = &cv.bias {
        bias_plane(c, &out, bias, cv.cout, n)?;
    }
    Ok(DActs { buf: out, c: cv.cout, h: src.h, w: src.w })
}

/// BatchNorm + ReLU in place, using the folded scale/shift.
fn bn_relu_inplace(x: &mut DActs, bn: &DBn) -> Result<(), String> {
    let c = ctx()?;
    let plane = x.plane() as i64;
    let mut aa = Args::new();
    aa.ptr(x.buf.ptr as u64);
    aa.ptr(bn.scale.ptr as u64);
    aa.ptr(bn.shift.ptr as u64);
    aa.i64(plane as i64);
    aa.i32(bn.channels as i32);
    c.launch_at(&mut aa, "k_bn_relu", (grid_for((x.plane() * bn.channels) as usize, BLOCK), 1, 1), (BLOCK, 1, 1))

}

/// BatchNorm + ReLU + residual add in one pass: `x = relu(scale*x + shift) + skip`.
/// Folding the residual add into the BN+ReLU removes one full pass over the
/// activation and one kernel launch per resblock, and keeps the block hot in L2
/// between the two.
fn bn_relu_add_inplace(x: &mut DActs, bn: &DBn, skip: Option<&DActs>) -> Result<(), String> {
    let c = ctx()?;
    let plane = x.plane() as i64;
    let kp = match skip {
        Some(s) => s.buf.ptr,
        None => ptr::null_mut(),
    };
    let mut aa = Args::new();
    aa.ptr(x.buf.ptr as u64);
    aa.ptr(kp as u64);
    aa.ptr(bn.scale.ptr as u64);
    aa.ptr(bn.shift.ptr as u64);
    aa.i64(plane as i64);
    aa.i32(bn.channels as i32);
    c.launch_at(&mut aa, "k_bn_relu_add", (grid_for((x.plane() * bn.channels) as usize, BLOCK), 1, 1), (BLOCK, 1, 1))

}

fn sigmoid_inplace(x: &mut DActs) -> Result<(), String> {
    let c = ctx()?;
    let mut aa = Args::new();
    // The toolkit's `lg_sigmoid` is out-of-place `(x, y, n)`, so in-place is the
    // same pointer twice. It computes `1/(1+__expf(-x))`, elementwise and
    // independently, so nothing is read after it is written - the local
    // `k_sigmoid` this replaces was that arithmetic exactly. The output layer is
    // the only caller, and its buffer is large (h*w per channel), so the
    // 1-D grid is over elements, not planes.
    aa.ptr(x.buf.ptr as u64);
    aa.ptr(x.buf.ptr as u64);
    aa.i64(x.buf.len as i64);
    c.launch_at(&mut aa, "lg_sigmoid", (grid_for(x.buf.len, BLOCK), 1, 1), (BLOCK, 1, 1))

}

fn add_inplace(a: &mut DActs, b: &DActs) -> Result<(), String> {
    let c = ctx()?;
    let mut aa = Args::new();
    aa.ptr(a.buf.ptr as u64);
    aa.ptr(b.buf.ptr as u64);
    aa.i64(a.buf.len as i64);
    c.launch_at(&mut aa, "lg_add_inplace", (grid_for(a.buf.len, BLOCK), 1, 1), (BLOCK, 1, 1))

}

/// ReflectionPad2d.
fn reflect_pad(src: &DActs, pad: usize) -> Result<DActs, String> {
    let c = ctx()?;
    let (oh, ow) = (src.h + 2 * pad, src.w + 2 * pad);
    let out = c.alloc(src.c * oh * ow)?;
    let mut aa = Args::new();
    aa.ptr(src.buf.ptr as u64);
    aa.ptr(out.ptr as u64);
    aa.i32(src.c as i32);
    aa.i32(src.h as i32);
    aa.i32(src.w as i32);
    aa.i32(pad as i32);
    c.launch_at(&mut aa, "k_reflect_pad", (grid_for(src.c * oh * ow, BLOCK), 1, 1), (BLOCK, 1, 1))?;
    Ok(DActs { buf: out, c: src.c, h: oh, w: ow })
}

fn avgpool2x2(src: &DActs) -> Result<DActs, String> {
    let c = ctx()?;
    let (oh, ow) = (src.h / 2, src.w / 2);
    let out = c.alloc(src.c * oh * ow)?;
    let mut aa = Args::new();
    aa.ptr(src.buf.ptr as u64);
    aa.ptr(out.ptr as u64);
    aa.i32(src.c as i32);
    aa.i32(src.h as i32);
    aa.i32(src.w as i32);
    c.launch_at(&mut aa, "k_avgpool2x2", (grid_for(src.c * oh * ow, BLOCK), 1, 1), (BLOCK, 1, 1))?;
    Ok(DActs { buf: out, c: src.c, h: oh, w: ow })
}

/// Concatenate two activation blocks along channels with a device kernel.
/// `cuMemcpyDtoD` is a blocking API call and therefore serialises the stream; a
/// copy kernel keeps everything ordered on the stream and costs one launch.
fn concat_acts(a: &DActs, g: &DActs) -> Result<DActs, String> {
    let c = ctx()?;
    let out = c.alloc(a.buf.len + g.buf.len)?;
    let mut off = 0usize;
    for src in [a, g] {
        let mut aa = Args::new();
        aa.ptr(src.buf.ptr as u64);
        aa.ptr((unsafe { (out.ptr as *mut u8).add(off) as *mut c_void }) as u64);
        aa.i64(src.buf.len as i64);
        c.launch_at(&mut aa, "lg_copy", (grid_for(src.buf.len, BLOCK), 1, 1), (BLOCK, 1, 1))?;
        off += src.buf.len * 4;
    }
    Ok(DActs { buf: out, c: a.c + g.c, h: a.h, w: a.w })
}

/// Transposed convolution through the four phase GEMMs.
/// For stride 2, pad 1 and a 3x3 kernel the output phase `(oy % 2, ox % 2)`
/// selects a fixed tap set (1, 2, 2 and 4 taps), so each phase is a dense
/// `[cout][cin*taps] x [cin*taps][n]` SGEMM over a gathered patch matrix - the
/// same im2col + GEMM shape as the forward convolutions, and much friendlier
/// to the memory system than the per-output gather kernel.
fn conv_transpose_phases(cv: &DConvT, src: &DActs, phases: &[Dev]) -> Result<DActs, String> {
    let c = ctx()?;
    let (oh, ow) = (src.h * 2, src.w * 2);
    let out = c.alloc(cv.cout * oh * ow)?;
    // No cudaMemset: each phase writes a disjoint strided subset of the output
    // pixels and `k_convt_put` writes (rather than accumulates) for the first
    // phase, so the buffer needs no pre-clearing - and cudaMemset is a blocking
    // API call that would serialise the stream anyway.
    // Per-phase output geometry: `ph_n` pixels per channel.
    for ph in 0..4 {
        let py = ph / 2;
        let px = ph % 2;
        let taps = CONVT_TAP_COUNT[ph];
        let k = cv.cin * taps;
        let n = ((oh + 1) / 2) * ((ow + 1) / 2);
        let col = c.alloc(k * n)?;
        let phase_out = c.alloc(cv.cout * n)?;
        let mut aa = Args::new();
        aa.ptr(src.buf.ptr as u64);
        aa.ptr(col.ptr as u64);
        aa.i32(cv.cin as i32);
        aa.i32(src.h as i32);
        aa.i32(src.w as i32);
        aa.i32(py as i32);
        aa.i32(px as i32);
        aa.i32(taps as i32);
        aa.i32(n as i32);
        aa.i32(oh as i32);
        aa.i32(ow as i32);
        c.launch_at(&mut aa, "k_convt_col", (grid_for(k * n, BLOCK), 1, 1), (BLOCK, 1, 1))?;

        sgemm(c, &col, &phases[ph], &phase_out, k, cv.cout, n)?;

        // `k_convt_put` always assigns (never accumulates) and every phase owns a
        // disjoint set of output pixels - phase (py, px) writes only the pixels
        // whose coordinates have those parities - so no output buffer clearing is
        // needed.  The flag only controls whether the bias is folded in.
        let mut aa = Args::new();
        aa.ptr(phase_out.ptr as u64);
        aa.ptr(out.ptr as u64);
        aa.ptr(cv.bias.ptr as u64);
        aa.i32(cv.cout as i32);
        aa.i32(n as i32);
        aa.i32(py as i32);
        aa.i32(px as i32);
        aa.i32(oh as i32);
        aa.i32(ow as i32);
        aa.i32(1i32 as i32);
        c.launch_at(&mut aa, "k_convt_put", (grid_for(cv.cout * n, BLOCK), 1, 1), (BLOCK, 1, 1))?;
    }
    Ok(DActs { buf: out, c: cv.cout, h: oh, w: ow })
}

fn conv_transpose(cv: &DConvT, src: &DActs) -> Result<DActs, String> {
    let c = ctx()?;
    let (oh, ow) = (src.h * 2, src.w * 2);
    let out = c.alloc(cv.cout * oh * ow)?;
    let mut aa = Args::new();
    aa.ptr(src.buf.ptr as u64);
    aa.ptr(cv.w.ptr as u64);
    aa.ptr(cv.bias.ptr as u64);
    aa.ptr(out.ptr as u64);
    aa.i32(cv.cin as i32);
    aa.i32(cv.cout as i32);
    aa.i32(src.h as i32);
    aa.i32(src.w as i32);
    aa.i32(oh as i32);
    aa.i32(ow as i32);
    aa.i32(3i32 as i32);
    aa.i32(3i32 as i32);
    aa.i32(2i32 as i32);
    aa.i32(1i32 as i32);
    c.launch_at(&mut aa, "k_conv_transpose", (grid_for(cv.cout * oh * ow, BLOCK), 1, 1), (BLOCK, 1, 1))?;
    Ok(DActs { buf: out, c: cv.cout, h: oh, w: ow })
}

// ---------------------------------------------------------------------------
// Step engine
// ---------------------------------------------------------------------------

/// One FFC_BN_ACT evaluation, mirroring `cpu::run_ffc` exactly.
/// `inp` arrives as `[in_local + in_global][h][w]` with the local half first;
/// the local half feeds `convl2l`/`convl2g` and the global half feeds
/// `convg2l`/`spectral`.  The two outputs are `(local, global)`.
fn run_dffc(
    w: &DFfc,
    inp: &DActs,
    glob: Option<&DActs>,
    skip_local: Option<&DActs>,
    skip_global: Option<&DActs>,
) -> Result<(DActs, Option<DActs>), String> {
    let local_view;
    let local = if w.in_global == 0 {
        inp
    } else {
        local_view = inp.view(0, w.in_local);
        &local_view
    };

    let out_local = if w.out_local > 0 {
        let mut acc = match &w.l2l {
            Some(cv) => conv_forward(cv, local, w.pad, w.stride, w.reflect)?,
            None => DActs::new(0, inp.h, inp.w)?,
        };
        if let Some(cv) = &w.g2l {
            let g = glob.ok_or("convg2l present without a global branch")?;
            let o = conv_forward(cv, g, w.pad, w.stride, w.reflect)?;
            if acc.c == 0 {
                acc = o;
            } else {
                add_inplace(&mut acc, &o)?;
            }
        }
        if let Some(bn) = &w.bn_l {
            // For the closing convolution of a resblock the residual add folds
            // into this BN+ReLU, saving a pass over the activation and a
            // launch; other FFC evaluations pass no skip.
            bn_relu_add_inplace(&mut acc, bn, skip_local)?;
        }
        Some(acc)
    } else {
        None
    };

    let out_global = if w.out_global > 0 {
        let mut acc = match &w.l2g {
            Some(cv) => conv_forward(cv, local, w.pad, w.stride, w.reflect)?,
            None => DActs::new(0, inp.h, inp.w)?,
        };
        if let Some(sp) = &w.spectral {
            let g = glob.ok_or("convg2g present without a global branch")?;
            // The spectral branch works on the global half only.
            let g_view = g.view(g.c - w.in_global, w.in_global);
            let o = spectral_forward(sp, &g_view)?;
            if acc.c == 0 {
                acc = o;
            } else {
                add_inplace(&mut acc, &o)?;
            }
        }
        if let Some(bn) = &w.bn_g {
            bn_relu_add_inplace(&mut acc, bn, skip_global)?;
        }
        Some(acc)
    } else {
        None
    };

    // An absent local branch still has to carry the plane geometry.
    let local_out = match out_local {
        Some(l) => l,
        None => DActs::new(0, inp.h, inp.w)?,
    };
    Ok((local_out, out_global))
}

fn copy_acts(src: &DActs) -> Result<DActs, String> {
    let c = ctx()?;
    let out = c.alloc(src.buf.len)?;
    let mut aa = Args::new();
    aa.ptr(src.buf.ptr as u64);
    aa.ptr(out.ptr as u64);
    aa.i64(src.buf.len as i64);
    c.launch_at(&mut aa, "lg_copy", (grid_for(src.buf.len, BLOCK), 1, 1), (BLOCK, 1, 1))?;
    Ok(DActs { buf: out, c: src.c, h: src.h, w: src.w })
}

/// The spectral branch, mirroring `cpu::Spectral::forward`.
/// The Fourier unit runs on the device through this engine's own batched 2-D
/// transforms (`lg_fft2_r2c` / `lg_fft2_c2r`).  `lg_fft2_r2c` produces the
/// half-spectrum `[half][h][w/2+1]` that PyTorch's `rfftn(norm='ortho')`
/// defines, and `lg_fft2_c2r` inverts it with the same Hermitian conventions
/// `irfft2_ortho` implements by hand (the imaginary parts of the first and last
/// columns are ignored, which is what makes the result real).  Both transforms
/// are unnormalised, so each half applies the `1/sqrt(h*w)` ortho factor where
/// the cuFFT pair used to: the forward through `k_spec_pack` and the inverse
/// through the `scale` argument of `lg_fft2_c2r` (which is why there is no
/// separate scaling launch).
/// Set `LAMA_FFT_HOST=1` to run the transforms on the CPU instead; the two paths
/// agree to float rounding and the host one is kept for cross-checking.
fn spectral_forward(sp: &DSpectral, g: &DActs) -> Result<DActs, String> {
    let c = ctx()?;
    let pooled = if sp.stride == 2 { avgpool2x2(g)? } else { copy_acts(g)? };
    let mut feat = conv_forward(&sp.conv1, &pooled, 0, 1, false)?;
    bn_relu_inplace(&mut feat, &sp.bn1)?;

    let (h, w) = (feat.h, feat.w);
    let half = sp.half;
    let hw = w / 2 + 1;
    let scale = 1.0f32 / ((h * w) as f32).sqrt();

    let t_fft = std::time::Instant::now();
    let spec_acts = if flag("LAMA_FFT_HOST") {
        let host = c.download(&feat.buf)?;
        let mut spec = vec![0f32; 2 * half * h * hw];
        for ch in 0..half {
            crate::cpu::rfft2_ortho(&host[ch * h * w..(ch + 1) * h * w], h, w, &mut spec[2 * ch * h * hw..]);
        }
        DActs { buf: c.upload(&spec)?, c: 2 * half, h, w: hw }
    } else {
        // Forward R2C into `[half][h][hw]` complex, then pack to the stacked
        // real/imag layout the 1x1 convolution consumes.
        let complex = c.alloc(2 * half * h * hw)?;
        let mut aa = Args::new();
        aa.ptr(feat.buf.ptr as u64);
        aa.ptr(complex.ptr as u64);
        aa.i32(half as i32);
        aa.i32(h as i32);
        aa.i32((h as f32).log2() as i32);
        aa.f32(1.0f32 as f32);
        c.launch_at(&mut aa, "lg_fft2_r2c", (half as u32, 1, 1), (256, 1, 1))?;
        let packed = c.alloc(2 * half * h * hw)?;
        let mut aa = Args::new();
        aa.ptr(complex.ptr as u64);
        aa.ptr(packed.ptr as u64);
        aa.i32(half as i32);
        aa.i32((h * hw) as i32);
        aa.f32(scale as f32);
        c.launch_at(&mut aa, "k_spec_pack", (grid_for(half * h * hw, BLOCK), 1, 1), (BLOCK, 1, 1))?;
        DActs { buf: packed, c: 2 * half, h, w: hw }
    };

    let mut fu = conv_forward(&sp.fu_conv, &spec_acts, 0, 1, false)?;
    bn_relu_inplace(&mut fu, &sp.fu_bn)?;

    let mut inv_acts = if flag("LAMA_FFT_HOST") {
        let fu_host = c.download(&fu.buf)?;
        let mut inv = vec![0f32; half * h * w];
        for ch in 0..half {
            crate::cpu::irfft2_ortho(&fu_host[2 * ch * h * hw..], h, w, &mut inv[ch * h * w..(ch + 1) * h * w]);
        }
        DActs { buf: c.upload(&inv)?, c: half, h, w }
    } else {
        // Unpack the stacked layout back to interleaved complex and run the
        // inverse C2R into its own real plane.  The transform is unnormalised
        // like the cuFFT one it replaced, so it takes the 1/sqrt(h*w) ortho
        // factor directly and no separate scaling launch is needed.
        let complex = c.alloc(2 * half * h * hw)?;
        let mut aa = Args::new();
        aa.ptr(fu.buf.ptr as u64);
        aa.ptr(complex.ptr as u64);
        aa.i32(half as i32);
        aa.i32((h * hw) as i32);
        c.launch_at(&mut aa, "k_spec_unpack", (grid_for(half * h * hw, BLOCK), 1, 1), (BLOCK, 1, 1))?;
        let real = c.alloc(half * h * w)?;
        let mut aa = Args::new();
        aa.ptr(complex.ptr as u64);
        aa.ptr(real.ptr as u64);
        aa.i32(half as i32);
        aa.i32(h as i32);
        aa.i32((h as f32).log2() as i32);
        aa.f32(scale as f32);
        c.launch_at(&mut aa, "lg_fft2_c2r", (half as u32, 1, 1), (256, 1, 1))?;
        DActs { buf: real, c: half, h, w }
    };
    fft_stat_add(t_fft.elapsed().as_secs_f32());
    // Add the pre-transform feature (the FourierUnit residual) and close with conv2.
    add_inplace(&mut inv_acts, &feat)?;
    conv_forward(&sp.conv2, &inv_acts, 0, 1, false)
}

/// The full forward pass, mirroring `cpu::Net::forward`.
pub fn run(
    store: &WeightStore,
    model: &Model,
    input: &[f32],
    h: usize,
    w: usize,
) -> Result<Vec<f32>, String> {
    let t_run = std::time::Instant::now();
    let t_ctx = std::time::Instant::now();
    let c = ctx()?;
    if flag("LAMA_PROFILE") {
        eprintln!("gpu ctx: {:.3}s", t_ctx.elapsed().as_secs_f32());
    }
    // Loading the weights onto the device is a one-off per process but it is a
    // cuMemcpyHtoD per tensor, and it happens inside this function, so the
    // caller's "inference" timing includes it while the per-step profile does
    // not.  Time it separately so the two can be told apart.
    let t_load = std::time::Instant::now();
    let steps = load_steps(store, model)?;
    if flag("LAMA_PROFILE") {
        eprintln!("gpu load_steps: {:.3}s", t_load.elapsed().as_secs_f32());
    }
    let t_up = std::time::Instant::now();
    let mut acts = DActs { buf: c.upload(input)?, c: 4, h, w };
    if flag("LAMA_PROFILE") {
        c.sync()?;
        eprintln!("gpu input upload: {:.3}s", t_up.elapsed().as_secs_f32());
    }
    let t_loop = std::time::Instant::now();
    let mut global: Option<DActs> = None;
    // `LAMA_DUMP_STEP=<n>` writes the same activation layout as the CPU path so
    // the two engines can be diffed step by step.
    let dump_after: Option<usize> = std::env::var("LAMA_DUMP_STEP").ok().and_then(|v| v.parse().ok());
    let mut step_no = 0usize;
    let profile = flag("LAMA_PROFILE");

    for step in &steps {
        let t_step = std::time::Instant::now();
        match step {
            DStep::ReflectPad(p) => {
                // Both branches carry spatial padding, exactly as the CPU path.
                acts = op_scope("pad", || reflect_pad(&acts, *p))?;
                global = match global {
                    Some(g) => Some(op_scope("pad", || reflect_pad(&g, *p))?),
                    None => None,
                };
            }
            DStep::Ffc(b) => {
                let (l, g) = op_scope("ffc", || run_dffc(b, &acts, global.as_ref(), None, None))?;
                acts = l;
                global = g;
            }
            DStep::Res(rb1, rb2) => {
                // conv1 then conv2, then the residual add on both branches,
                // exactly as `cpu.rs` does it.  Neither `run_dffc` writes through
                // its input, so the pre-resblock activation is still intact and
                // can serve as the skip directly - no copy and no separate add
                // pass: the add folds into conv2's closing BN+ReLU.
                let (l1, g1) = op_scope("res_a", || run_dffc(rb1, &acts, global.as_ref(), None, None))?;
                let (l2, g2) = op_scope("res_b", || {
                    run_dffc(rb2, &l1, g1.as_ref(), Some(&acts), global.as_ref())
                })?;
                acts = l2;
                global = g2;
            }
            DStep::Concat => {
                // local | global stacked along channels, copied on the device so
                // the stream is never blocked by a host-side copy.
                let g = global.take().ok_or("Concat without a global branch")?;
                acts = concat_acts(&acts, &g)?;
            }
            DStep::Upsample(convt, bn) => {
                // The four phase GEMMs are the fast path; `LAMA_CONVT_GATHER=1`
                // selects the reference gather kernel for cross-checking.
                let mut up = op_scope("upsample", || match &convt.phases {
                    Some(ph) if !flag("LAMA_CONVT_GATHER") => conv_transpose_phases(convt, &acts, ph),
                    _ => conv_transpose(convt, &acts),
                })?;
                bn_relu_inplace(&mut up, bn)?;
                acts = up;
                // The upsample collapses the two branches into one.
                global = None;
            }
            DStep::OutConv(cv) => {
                // The plan emits ReflectPad(3) immediately before this step, so
                // the 7x7 convolution itself adds no padding (pad 0).
                let mut out = op_scope("outconv", || conv_forward(cv, &acts, 0, 1, false))?;
                sigmoid_inplace(&mut out)?;
                c.sync()?;
                if profile {
                    eprintln!("gpu step loop: {:.3}s", t_loop.elapsed().as_secs_f32());
                }
                let t_dl = std::time::Instant::now();
                let r = c.download(&out.buf);
                if profile {
                    eprintln!(
                        "gpu output download: {:.3}s ({} floats)",
                        t_dl.elapsed().as_secs_f32(),
                        out.buf.len
                    );
                    eprintln!("gpu run total: {:.3}s", t_run.elapsed().as_secs_f32());
                }
                return r;
            }
        }
        if dump_after == Some(step_no) {
            c.sync()?;
            let mut all = c.download(&acts.buf)?;
            if let Some(g) = &global {
                all.extend_from_slice(&c.download(&g.buf)?);
            }
            let mut bytes = Vec::with_capacity(all.len() * 4);
            for v in &all {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            let _ = std::fs::write(format!("/tmp/gpu_step_{step_no}.f32"), &bytes);
            eprintln!(
                "gpu dumped step {step_no}: local {}x{}x{} global {:?}",
                acts.c, acts.h, acts.w, global.as_ref().map(|g| (g.c, g.h, g.w))
            );
        }
        if flag("LAMA_PROFILE_SYNC") {
            // The per-step figure above only measures *submission*: every kernel
            // launch is asynchronous, so the timings say how long the host took
            // to queue the work, not how long the device took to run it.  With a
            // sync at the end of the step the figure becomes submit + device,
            // and the difference between the two is the device time.
            c.sync()?;
        }
        if profile {
            let (nfft, t_fft) = FFT_STATS.with(|s| {
                let s = s.borrow();
                (s.0, s.1)
            });
            eprintln!(
                "gpu step {step_no:2}: {:>6.3}s local {}x{}x{} global {:?}  [fft calls {} {:.3}s]",
                t_step.elapsed().as_secs_f32(),
                acts.c,
                acts.h,
                acts.w,
                global.as_ref().map(|g| (g.c, g.h, g.w)),
                nfft,
                t_fft,
            );
        }
        step_no += 1;
    }
    c.sync()?;
    Ok(c.download(&acts.buf)?)
}

// (spectral host-FFT calls, seconds spent in the device<->host round trip).
thread_local! {
    static FFT_STATS: std::cell::RefCell<(usize, f32)> = const { std::cell::RefCell::new((0, 0.0)) };
}

// (cuMemAlloc calls, seconds).
thread_local! {
    static ALLOC_STATS: std::cell::RefCell<(usize, f32)> = const { std::cell::RefCell::new((0, 0.0)) };
}

/// Time one step kind when `LAMA_PROFILE_OPS` is set.  Each call synchronises,
/// so the totals are attributable kernel time at the price of removing all
/// overlap, which the per-step numbers alone are not.
fn op_scope<T>(kind: &'static str, f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    if !flag("LAMA_PROFILE_OPS") {
        return f();
    }
    let c = match Cuda::get() {
        Some(c) => c,
        None => return f(),
    };
    c.sync()?;
    let n0 = LAUNCHES.with(|c| c.get());
    let t = std::time::Instant::now();
    let out = f()?;
    c.sync()?;
    let n1 = LAUNCHES.with(|c| c.get());
    OP_STATS.with(|s| {
        let mut m = s.borrow_mut();
        let e = m.entry(kind).or_insert((0usize, 0.0f32, 0usize));
        e.0 += 1;
        e.1 += t.elapsed().as_secs_f32();
        e.2 += n1 - n0;
    });
    Ok(out)
}

// (sub-op) -> (calls, seconds, output elements, FLOPs) for the convolution
// internals, so a profiled pass says whether im2col or the GEMM dominates.
thread_local! {
    static SUB_STATS: std::cell::RefCell<std::collections::HashMap<&'static str, (usize, f32, usize, usize)>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Print the convolution sub-step breakdown when `LAMA_PROFILE_SUB` is set.
pub fn print_sub_stats() {
    if !flag("LAMA_PROFILE_SUB") {
        return;
    }
    let mut v: Vec<(&str, (usize, f32, usize, usize))> =
        SUB_STATS.with(|s| s.borrow().iter().map(|(k, x)| (*k, *x)).collect());
    v.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
    for (k, (n, t, elems, flops)) in &v {
        eprintln!(
            "sub {k:9} {n:4} calls {:>7.3}s  ({} us avg)  {:.1} MFLOP  {:.0} GFLOP/s  {:.0} MB moved",
            t,
            (*t * 1e6 / *n as f32) as i64,
            *flops as f64 / 1e6,
            *flops as f64 / 1e9 / (*t as f64).max(1e-9),
            elems * 4 / 1_000_000
        );
    }

}

// (op kind) -> (calls, seconds, launches).
thread_local! {
    static OP_STATS: std::cell::RefCell<std::collections::HashMap<&'static str, (usize, f32, usize)>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Print the per-op-kind counters when `LAMA_PROFILE_OPS` is set.
pub fn print_op_stats() {
    if !flag("LAMA_PROFILE_OPS") {
        return;
    }
    let mut v: Vec<(&str, (usize, f32, usize))> =
        OP_STATS.with(|s| s.borrow().iter().map(|(k, x)| (*k, *x)).collect());
    v.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
    let total: f32 = v.iter().map(|(_, x)| x.1).sum();
    for (k, (n, t, l)) in &v {
        eprintln!(
            "op {k:14} {n:5} calls {:>7.3}s  ({} us avg, {} launches/call)",
            t,
            (*t * 1e6 / *n as f32) as i64,
            *l / *n
        );
    }
    eprintln!("op total {total:.3}s");
}

/// Print the allocation and spectral counters when `LAMA_PROFILE` is set.
pub fn print_stats() {
    if !flag("LAMA_PROFILE") {
        return;
    }
    let (na, ta) = ALLOC_STATS.with(|s| {
        let s = s.borrow();
        (s.0, s.1)
    });
    let (nf, tf) = FFT_STATS.with(|s| {
        let s = s.borrow();
        (s.0, s.1)
    });
    let (hits, misses, cached) = POOL.with(|s| {
        let s = s.borrow();
        (s.hits, s.misses, s.bytes)
    });
    eprintln!(
        "gpu totals: {na} cuMemAlloc in {ta:.3}s; {nf} spectral FFT round trips in {tf:.3}s; \
         pool {hits} hits / {misses} misses, {} MB cached",
        cached >> 20
    );
}

fn fft_stat_add(secs: f32) {
    FFT_STATS.with(|s| {
        let mut s = s.borrow_mut();
        s.0 += 1;
        s.1 += secs;
    });
}
