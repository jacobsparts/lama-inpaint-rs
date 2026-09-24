//! CPU forward pass: direct convolutions over NCHW planes.
//!
//! This is the fallback path (and the correctness reference for the GPU path).
//! It runs one layer at a time over `Vec<f32>` buffers, parallelised with
//! rayon across output channels / rows.  No BLAS is required.

use crate::image::reflect_index;
use crate::model::{ConvRef, FfcBlock, Model, Step};
use crate::weights::WeightStore;
use rayon::prelude::*;
use std::time::Instant;

/// A batch of feature planes: `[c][h][w]`.
#[derive(Clone)]
pub struct Acts {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Acts {
    pub fn new(c: usize, h: usize, w: usize) -> Self {
        Self { c, h, w, data: vec![0f32; c * h * w] }
    }
    pub fn plane(&self, c: usize) -> &[f32] {
        &self.data[c * self.h * self.w..(c + 1) * self.h * self.w]
    }
    /// Metadata-only copy: same shape, empty data.  Used when an FFC branch is
    /// absent and only the plane geometry matters.
    pub fn clone_meta(&self) -> Self {
        Acts { c: self.c, h: self.h, w: self.w, data: self.data.clone() }
    }
}

/// BatchNorm2d in eval mode: `(x - mean) / sqrt(var + eps) * w + b`, folded into
/// a per-channel scale and shift up front.
pub struct Bn {
    pub scale: Vec<f32>,
    pub shift: Vec<f32>,
}

impl Bn {
    pub fn load(store: &WeightStore, prefix: &str, channels: usize) -> Result<Self, String> {
        let w = store.f32(&format!("{prefix}.weight"))?;
        let b = store.f32(&format!("{prefix}.bias"))?;
        let mean = store.f32(&format!("{prefix}.running_mean"))?;
        let var = store.f32(&format!("{prefix}.running_var"))?;
        if w.len() != channels || b.len() != channels || mean.len() != channels || var.len() != channels {
            return Err(format!(
                "BatchNorm {prefix}: expected {channels} channels, got w={} b={} mean={} var={}",
                w.len(), b.len(), mean.len(), var.len()
            ));
        }
        let mut scale = vec![0f32; channels];
        let mut shift = vec![0f32; channels];
        for i in 0..channels {
            let s = w[i] / (var[i] + 1e-5).sqrt();
            scale[i] = s;
            shift[i] = b[i] - mean[i] * s;
        }
        Ok(Bn { scale, shift })
    }

    pub fn apply(&self, acts: &mut Acts) {
        let plane = acts.h * acts.w;
        acts.data
            .par_chunks_mut(plane)
            .enumerate()
            .for_each(|(ci, p)| {
                let s = self.scale[ci];
                let t = self.shift[ci];
                p.iter_mut().for_each(|v| *v = *v * s + t);
            });
    }
}

/// A convolution weight packed as `[cout][cin][kh][kw]`, as stored.
pub struct Conv2dW {
    pub w: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub cin: usize,
    pub cout: usize,
    pub kh: usize,
    pub kw: usize,
}

impl Conv2dW {
    pub fn load(store: &WeightStore, prefix: &str) -> Result<Self, String> {
        let info = store.info(&format!("{prefix}.weight"))?.clone();
        let w = store.f32(&format!("{prefix}.weight"))?;
        let cout = info.shape[0];
        let cin = info.shape[1];
        let kh = info.shape[2];
        let kw = info.shape[3];
        let bias = if store.info(&format!("{prefix}.bias")).is_ok() {
            Some(store.f32(&format!("{prefix}.bias"))?)
        } else {
            None
        };
        Ok(Conv2dW { w, bias, cin, cout, kh, kw })
    }

    /// kxk convolution, stride `stride`, padding either zeros or reflection.
    ///
    /// PyTorch's `padding_mode='reflect'` with `padding=p` pre-pads p pixels by
    /// reflection and then runs a valid convolution, so the reflection only
    /// affects out-of-range taps and never changes the output size.
    pub fn forward_pad(&self, src: &Acts, pad: usize, stride: usize, reflect: bool) -> Acts {
        let (h, w) = (src.h, src.w);
        let oh = (h + 2 * pad - self.kh) / stride + 1;
        let ow = (w + 2 * pad - self.kw) / stride + 1;
        let mut out = Acts::new(self.cout, oh, ow);
        let (k, cin) = (self.kh * self.kw, self.cin);
        // Parallelising over output channels only gives 128-384 work items, each of
        // which is a full plane; splitting by (channel, output row) gives thousands
        // of items of uniform size, which rayon balances far better and keeps every
        // thread's writes inside its own row.
        out.data
            .par_chunks_mut(ow)
            .enumerate()
            .for_each(|(row, dst)| {
                let co = row / oh;
                let oy = row % oh;
                let wp = &self.w[co * cin * k..(co + 1) * cin * k];
                let b = self.bias.as_ref().map_or(0.0, |b| b[co]);
                for ox in 0..ow {
                    let mut acc = b;
                    for ci in 0..cin {
                        let sp = &src.data[ci * h * w..(ci + 1) * h * w];
                        let kp = &wp[ci * k..(ci + 1) * k];
                        for ky in 0..self.kh {
                            let iy = oy * stride + ky;
                            let sy = if iy < pad {
                                if !reflect {
                                    continue;
                                }
                                reflect_index(iy as isize - pad as isize, h)
                            } else if iy - pad >= h {
                                if !reflect {
                                    continue;
                                }
                                reflect_index(iy as isize - pad as isize, h)
                            } else {
                                iy - pad
                            };
                            let srow = &sp[sy * w..sy * w + w];
                            let krow = &kp[ky * self.kw..(ky + 1) * self.kw];
                            for kx in 0..self.kw {
                                let ix = ox * stride + kx;
                                let sx = if ix < pad || ix - pad >= w {
                                    if !reflect {
                                        continue;
                                    }
                                    reflect_index(ix as isize - pad as isize, w)
                                } else {
                                    ix - pad
                                };
                                acc += krow[kx] * srow[sx];
                            }
                        }
                    }
                    dst[ox] = acc;
                }
            });
        out
    }

    /// 1x1 convolution expressed as a channel matmul over `[cin][h][w]` planes.
    pub fn forward_1x1(&self, src: &Acts) -> Acts {
        debug_assert_eq!(self.kh, 1);
        debug_assert_eq!(self.kw, 1);
        let plane = src.h * src.w;
        let mut out = Acts::new(self.cout, src.h, src.w);
        out.data
            .par_chunks_mut(plane)
            .enumerate()
            .for_each(|(co, dst)| {
                let wp = &self.w[co * self.cin..(co + 1) * self.cin];
                let b = self.bias.as_ref().map_or(0.0, |b| b[co]);
                for i in 0..plane {
                    let mut acc = b;
                    for ci in 0..self.cin {
                        acc += wp[ci] * src.data[ci * plane + i];
                    }
                    dst[i] = acc;
                }
            });
        out
    }
}

/// Spectral transform (`SpectralTransform` with `enable_lfu=True`):
/// `conv2(conv1(x) + fourier_unit(conv1(x)))`, preceded by a strided average
/// pool and followed by a 1x1 convolution.
pub struct Spectral {
    pub stride: usize,
    pub conv1: Conv2dW, // 1x1, cin -> cout/2
    pub conv2: Conv2dW, // 1x1, cout/2 -> cout
    pub bn1: Bn,
    pub fu: FourierUnit,
    pub half: usize,
    /// `model.<index>` and FFC sub-index of the owning block, used only to tag
    /// debug dumps - a resblock's `conv1` and `conv2` share the model index.
    pub block: usize,
    pub sub: usize,
}

pub struct FourierUnit {
    /// 1x1 conv over the stacked real/imag channels: 2*half -> 2*half.
    pub conv: Conv2dW,
    pub bn: Bn,
    pub half: usize,
}

impl Spectral {
    pub fn load(
        store: &WeightStore,
        prefix: &str,
        cout: usize,
        stride: usize,
        block: usize,
        sub: usize,
    ) -> Result<Self, String> {
        let half = cout / 2;
        let conv1 = Conv2dW::load(store, &format!("{prefix}.conv1.0"))?;
        let bn1 = Bn::load(store, &format!("{prefix}.conv1.1"), half)?;
        let conv2 = Conv2dW::load(store, &format!("{prefix}.conv2"))?;
        let fu = FourierUnit {
            conv: Conv2dW::load(store, &format!("{prefix}.fu.conv_layer"))?,
            bn: Bn::load(store, &format!("{prefix}.fu.bn"), 2 * half)?,
            half,
        };
        Ok(Spectral { stride, conv1, conv2, bn1, fu, half, block, sub })
    }

    /// `x`: `[half][h][w]`; returns `[cout][h/stride][w/stride]`.
    pub fn forward(&self, x: &Acts) -> Acts {
        // Tagged with the owning block so a run that touches every spectral
        // layer does not overwrite the one stage under investigation.
        let tag = std::env::var("LAMA_DUMP_SPECTRAL")
            .ok()
            .map(|t| format!("{t}_{}_{}", self.block, self.sub));
        if let Some(t) = &tag {
            dump_acts(&format!("/tmp/rust_spec_{t}_in.f32"), &x.data);
        }
        let pooled = if self.stride == 2 { avg_pool2x2(x) } else { x.clone() };
        let mut feat = self.conv1.forward_1x1(&pooled);
        self.bn1.apply(&mut feat);
        relu_inplace(&mut feat);
        if let Some(t) = &tag {
            dump_acts(&format!("/tmp/rust_spec_{t}_conv1.f32"), &feat.data);
        }
        let fu = self.fu.forward(&feat, tag.as_deref());
        if let Some(t) = &tag {
            dump_acts(&format!("/tmp/rust_spec_{t}_fu.f32"), &fu.data);
        }
        let mut sum = Acts::new(self.half, feat.h, feat.w);
        sum.data
            .par_iter_mut()
            .zip(feat.data.par_iter().zip(fu.data.par_iter()))
            .for_each(|(s, (a, b))| *s = a + b);
        self.conv2.forward_1x1(&sum)
    }
}

impl FourierUnit {
    /// `x`: `[half][h][w]`; returns the inverse-transformed `[half][h][w]`.
    pub fn forward(&self, x: &Acts, tag: Option<&str>) -> Acts {
        let (h, w) = (x.h, x.w);
        let hw = w / 2 + 1;
        // `rfftn(norm='ortho')` per channel; real and imaginary parts are
        // stacked into two planes per channel, exactly as the reference does
        // before its 1x1 convolution.
        let prof = std::env::var("LAMA_PROFILE_FFT").is_ok();
        let t_fft = Instant::now();
        let mut spec = vec![0f32; 2 * self.half * h * hw];
        // Each channel is an independent 2-D transform, so the parallelism goes
        // here rather than deeper inside `fft2`.
        spec.par_chunks_mut(2 * h * hw)
            .enumerate()
            .for_each(|(c, dst)| {
                rfft2_ortho(x.plane(c), h, w, dst);
            });
        if prof {
            eprintln!("  fu rfft {:.3}s", t_fft.elapsed().as_secs_f32());
        }
        let t_conv = Instant::now();
        let s = Acts { c: 2 * self.half, h, w: hw, data: spec };
        if let Some(t) = tag {
            dump_acts(&format!("/tmp/rust_spec_{t}_spec.f32"), &s.data);
        }
        let mut conv = self.conv.forward_1x1(&s);
        self.bn.apply(&mut conv);
        relu_inplace(&mut conv);
        if prof {
            eprintln!("  fu conv+bn {:.3}s", t_conv.elapsed().as_secs_f32());
        }
        let mut out = Acts::new(self.half, h, w);
        out.data
            .par_chunks_mut(h * w)
            .enumerate()
            .for_each(|(c, dst)| {
                irfft2_ortho(&conv.data[2 * c * h * hw..], h, w, dst);
            });
        out
    }
}

/// Transposed convolution `[cin][h][w] -> [cout][2h][2w]` for k3 s2 p1 op1.
#[allow(dead_code)]  // `cin` mirrors the reference description.
pub struct ConvT {
    pub w: Vec<f32>, // [cin][cout][3][3]
    pub bias: Vec<f32>,
    pub cin: usize,
    pub cout: usize,
}

impl ConvT {
    pub fn load(store: &WeightStore, prefix: &str) -> Result<Self, String> {
        let info = store.info(&format!("{prefix}.weight"))?.clone();
        let w = store.f32(&format!("{prefix}.weight"))?;
        let bias = store.f32(&format!("{prefix}.bias"))?;
        Ok(ConvT { w, bias, cin: info.shape[0], cout: info.shape[1] })
    }

    pub fn forward(&self, src: &Acts) -> Acts {
        let oh = src.h * 2;
        let ow = src.w * 2;
        let mut out = Acts::new(self.cout, oh, ow);
        // Scatter each input pixel's 3x3 contribution.  Parallelising over
        // output channels keeps each thread's slab private, so no reduction or
        // synchronisation is needed; the scatter touches at most 9 outputs per
        // input pixel (weights are tiny next to the activations).
        out.data
            .par_chunks_mut(oh * ow)
            .enumerate()
            .for_each(|(co, dst)| {
                let b = self.bias[co];
                for v in dst.iter_mut() {
                    *v = b;
                }
                for ci in 0..src.c {
                    let sp = src.plane(ci);
                    let kp = &self.w[(ci * self.cout + co) * 9..][..9];
                    for iy in 0..src.h {
                        let row = &sp[iy * src.w..(iy + 1) * src.w];
                        for (ix, v) in row.iter().enumerate() {
                            if *v == 0.0 {
                                continue;
                            }
                            for ky in 0..3 {
                                let oy = iy * 2 + ky;
                                // padding=1: output row oy-1 must be in range.
                                if oy == 0 || oy > oh {
                                    continue;
                                }
                                let dst_row = &mut dst[(oy - 1) * ow..oy * ow];
                                for kx in 0..3 {
                                    let ox = ix * 2 + kx;
                                    if ox == 0 || ox > ow {
                                        continue;
                                    }
                                    dst_row[ox - 1] += *v * kp[ky * 3 + kx];
                                }
                            }
                        }
                    }
                }
            });
        out
    }
}

pub fn relu_inplace(a: &mut Acts) {
    a.data.par_iter_mut().for_each(|v| {
        if *v < 0.0 {
            *v = 0.0;
        }
    });
}

pub fn sigmoid_inplace(a: &mut Acts) {
    a.data.par_iter_mut().for_each(|v| {
        *v = 1.0 / (1.0 + (-*v).exp());
    });
}

/// 2x2 average pool with stride 2 (the reference `AvgPool2d(kernel=2, stride=2)`).
fn avg_pool2x2(x: &Acts) -> Acts {
    let oh = x.h / 2;
    let ow = x.w / 2;
    let mut out = Acts::new(x.c, oh, ow);
    out.data
        .par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(c, dst)| {
            let sp = x.plane(c);
            for oy in 0..oh {
                for ox in 0..ow {
                    let a = sp[(2 * oy) * x.w + 2 * ox];
                    let b = sp[(2 * oy) * x.w + 2 * ox + 1];
                    let cc = sp[(2 * oy + 1) * x.w + 2 * ox];
                    let d = sp[(2 * oy + 1) * x.w + 2 * ox + 1];
                    dst[oy * ow + ox] = (a + b + cc + d) * 0.25;
                }
            }
        });
    out
}

// ------------------------------------------------------------------- the net

/// One FFC block with its parameters resolved.
#[allow(dead_code)]  // Geometry fields describe the block; not all are read here.
struct FfcWeights {
    kernel: usize,
    stride: usize,
    pad: usize,
    reflect: bool,
    in_local: usize,
    in_global: usize,
    out_local: usize,
    out_global: usize,
    l2l: Option<Conv2dW>,
    l2g: Option<Conv2dW>,
    g2l: Option<Conv2dW>,
    spectral: Option<Spectral>,
    bn_l: Option<Bn>,
    bn_g: Option<Bn>,
}

struct ResWeights {
    conv1: FfcWeights,
    conv2: FfcWeights,
}

struct UpsampleWeights {
    convt: ConvT,
    bn: Bn,
}

enum NetStep {
    ReflectPad(usize),
    Ffc(FfcWeights),
    Res(ResWeights),
    Concat,
    Upsample(UpsampleWeights),
    OutConv(Conv2dW),
}

/// The generator with every weight resolved.  Loading is separated from
/// execution so the one-shot process touches the blob exactly once.
pub struct Net {
    steps: Vec<NetStep>,
}

impl Net {
    pub fn load(store: &WeightStore, model: &Model) -> Result<Self, String> {
        let mut steps = Vec::with_capacity(model.steps.len());
        for step in &model.steps {
            match step {
                Step::ReflectPad(p) => steps.push(NetStep::ReflectPad(*p)),
                Step::Ffc(b) => steps.push(NetStep::Ffc(load_ffc(store, b)?)),
                Step::ResBlock(rb) => steps.push(NetStep::Res(ResWeights {
                    conv1: load_ffc(store, &rb.conv1)?,
                    conv2: load_ffc(store, &rb.conv2)?,
                })),
                Step::Concat => steps.push(NetStep::Concat),
                Step::Upsample(u) => steps.push(NetStep::Upsample(UpsampleWeights {
                    convt: ConvT::load(store, &format!("model.{}", u.index))?,
                    bn: Bn::load(store, &format!("model.{}", u.bn_index), u.cout)?,
                })),
                Step::OutConv(o) => steps.push(NetStep::OutConv(Conv2dW::load(
                    store,
                    &format!("model.{}", o.index),
                )?)),
            }
        }
        Ok(Net { steps })
    }

    /// Run the network.  `input` is `[4][h][w]`; the result is `[3][h][w]`.
    pub fn forward(&self, input: Acts) -> Acts {
        let mut local = input;
        let mut global: Option<Acts> = None;
        // `LAMA_DUMP_STEP=<n>` writes the n-th intermediate activation next to
        // the local and global branches, so a mismatch against the torch
        // reference can be traced to the first divergent layer.
        let dump_after: Option<usize> = std::env::var("LAMA_DUMP_STEP").ok().and_then(|v| v.parse().ok());
        let mut step_no = 0usize;
        let profile = std::env::var("LAMA_PROFILE").is_ok();
        for step in &self.steps {
            let t = Instant::now();
            match step {
                NetStep::ReflectPad(pad) => {
                    local = reflect_pad_acts(&local, *pad);
                    if let Some(g) = &global {
                        global = Some(reflect_pad_acts(g, *pad));
                    }
                }
                NetStep::Ffc(w) => {
                    let (nl, ng) = run_ffc(w, &local, global.as_ref());
                    local = nl;
                    global = ng;
                }
                NetStep::Res(rw) => {
                    let (nl, ng) = run_ffc(&rw.conv1, &local, global.as_ref());
                    let (nl2, ng2) = run_ffc(&rw.conv2, &nl, ng.as_ref());
                    // Residual connections on both branches.
                    add_inplace(&mut local, &nl2);
                    match (&mut global, ng2) {
                        (Some(g), Some(n)) => add_inplace(g, &n),
                        (None, Some(n)) => global = Some(n),
                        _ => {}
                    }
                }
                NetStep::Concat => {
                    if let Some(g) = global.take() {
                        local = concat(&local, &g);
                    }
                }
                NetStep::Upsample(u) => {
                    let mut a = u.convt.forward(&local);
                    u.bn.apply(&mut a);
                    relu_inplace(&mut a);
                    local = a;
                    global = None;
                }
                NetStep::OutConv(cv) => {
                    local = cv.forward_pad(&local, 0, 1, false);
                    sigmoid_inplace(&mut local);
                }
            }
            if dump_after == Some(step_no) {
                let mut all = local.data.clone();
                if let Some(g) = &global {
                    all.extend_from_slice(&g.data);
                }
                dump_acts(&format!("/tmp/rust_step_{step_no}.f32"), &all);
                eprintln!(
                    "dumped step {step_no}: local {}x{}x{} global {:?}",
                    local.c, local.h, local.w, global.as_ref().map(|g| (g.c, g.h, g.w))
                );
            }
            if profile {
                eprintln!(
                    "step {step_no:2}: {:>7.2}s  local {:?}{}",
                    t.elapsed().as_secs_f32(),
                    (local.c, local.h, local.w),
                    global.as_ref().map_or(String::new(), |g| format!(" global {:?}", (g.c, g.h, g.w)))
                );
            }
            step_no += 1;
        }
        local
    }
}

fn load_ffc(store: &WeightStore, b: &FfcBlock) -> Result<FfcWeights, String> {
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
    let load_conv = |r: &ConvRef| -> Result<Option<Conv2dW>, String> {
        match r {
            ConvRef::Absent => Ok(None),
            ConvRef::Present { name, .. } => {
                Ok(Some(Conv2dW::load(store, &format!("{ffc_prefix}.{name}"))?))
            }
        }
    };
    Ok(FfcWeights {
        kernel: b.kernel,
        stride: b.stride,
        pad: b.pad,
        reflect: b.reflection,
        in_local: b.in_local,
        in_global: b.in_global,
        out_local: b.out_local,
        out_global: b.out_global,
        l2l: load_conv(&b.l2l)?,
        l2g: load_conv(&b.l2g)?,
        g2l: load_conv(&b.g2l)?,
        spectral: if b.g2g {
            Some(Spectral::load(
                store,
                &format!("{ffc_prefix}.convg2g"),
                b.out_global,
                b.stride,
                b.index,
                b.sub,
            )?)
        } else {
            None
        },
        bn_l: if b.bn_local {
            Some(Bn::load(store, &format!("{partial}.bn_l"), b.out_local)?)
        } else {
            None
        },
        bn_g: if b.bn_global {
            Some(Bn::load(store, &format!("{partial}.bn_g"), b.out_global)?)
        } else {
            None
        },
    })
}

impl FfcWeights {
    #[inline]
    /// Run one of the block's spatial convolutions.  `pad` comes from the
    /// block description (0 for `model.1`, `kernel / 2` for the rest) and
    /// `reflect` selects PyTorch's `padding_mode` behaviour.
    fn conv(&self, w: &Conv2dW, src: &Acts) -> Acts {
        if self.kernel == 1 {
            w.forward_1x1(src)
        } else {
            w.forward_pad(src, self.pad, self.stride, self.reflect)
        }
    }
}

/// `FFC.forward`: local output is `convl2l(x_l) + convg2l(x_g)`, global output is
/// `convl2g(x_l) + convg2g(x_g)`, each followed by BatchNorm + ReLU when the
/// corresponding `ratio` is not 0 or 1.
fn run_ffc(w: &FfcWeights, local: &Acts, global: Option<&Acts>) -> (Acts, Option<Acts>) {
    let out_local = if w.out_local > 0 {
        let mut acc = match &w.l2l {
            Some(cv) => w.conv(cv, local),
            None => Acts::new(0, local.h, local.w),
        };
        if let Some(cv) = &w.g2l {
            let g = global.expect("convg2l present without a global branch");
            let o = w.conv(cv, g);
            if acc.c == 0 {
                acc = o;
            } else {
                add_inplace(&mut acc, &o);
            }
        }
        if let Some(bn) = &w.bn_l {
            bn.apply(&mut acc);
            relu_inplace(&mut acc);
        }
        Some(acc)
    } else {
        None
    };

    let out_global = if w.out_global > 0 {
        let mut acc = match &w.l2g {
            Some(cv) => w.conv(cv, local),
            None => Acts::new(0, local.h, local.w),
        };
        if let Some(sp) = &w.spectral {
            let g = global.expect("convg2g present without a global branch");
            let o = sp.forward(g);
            if acc.c == 0 {
                acc = o;
            } else {
                add_inplace(&mut acc, &o);
            }
        }
        if let Some(bn) = &w.bn_g {
            bn.apply(&mut acc);
            relu_inplace(&mut acc);
        }
        Some(acc)
    } else {
        None
    };

    let local_out = match (&out_local, &out_global) {
        (Some(l), _) => l.clone_meta(),
        (None, Some(g)) => Acts { c: 0, h: g.h, w: g.w, data: Vec::new() },
        (None, None) => Acts { c: 0, h: local.h, w: local.w, data: Vec::new() },
    };
    (local_out, out_global)
}

/// Write raw FP32 planes for offline comparison with the torch reference.
fn dump_acts(path: &str, data: &[f32]) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::File::create(path) {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let _ = f.write_all(&bytes);
    }
}

fn add_inplace(dst: &mut Acts, src: &Acts) {
    debug_assert_eq!(dst.data.len(), src.data.len());
    dst.data
        .par_iter_mut()
        .zip(src.data.par_iter())
        .for_each(|(a, b)| *a += *b);
}

/// Concatenate local and global branches along channels.
fn concat(a: &Acts, b: &Acts) -> Acts {
    let mut out = Acts::new(a.c + b.c, a.h, a.w);
    out.data[..a.data.len()].copy_from_slice(&a.data);
    out.data[a.data.len()..a.data.len() + b.data.len()].copy_from_slice(&b.data);
    out
}

fn reflect_pad_acts(a: &Acts, pad: usize) -> Acts {
    if pad == 0 {
        return a.clone();
    }
    let mut out = Acts::new(a.c, a.h + 2 * pad, a.w + 2 * pad);
    let (oh, ow) = (out.h, out.w);
    out.data
        .par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(c, dst)| {
            let sp = a.plane(c);
            for y in 0..oh {
                let sy = reflect_index(y as isize - pad as isize, a.h);
                for x in 0..ow {
                    let sx = reflect_index(x as isize - pad as isize, a.w);
                    dst[y * ow + x] = sp[sy * a.w + sx];
                }
            }
        });
    out
}

/// Entry point used by `main`: load the net, run one image, return `[3][h][w]`.
pub fn run(
    store: &WeightStore,
    model: &Model,
    input: Vec<f32>,
    h: usize,
    w: usize,
) -> Result<Vec<f32>, String> {
    let net = Net::load(store, model)?;
    let acts = Acts { c: 4, h, w, data: input };
    Ok(net.forward(acts).data)
}

// ------------------------------------------------------------------ FFT
//
// `FourierUnit` calls `rfftn(norm='ortho')` and `irfftn(s=input_shape,
// norm='ortho')`.  Torch's `norm='ortho'` scales by `1/sqrt(N)` on the forward
// and by `1/sqrt(N)` on the inverse, N being the transform size (h*w here) -
// so the composite is an exact inverse-preserving unitary pair.
//
// The 2-D transform is realised as row transforms then column transforms.  A
// plane can be any size (the network's planes are input/8 at the bottleneck),
// so lengths are handled by a radix-2 stage where possible and Bluestein's
// algorithm otherwise; correctness matters far more than the constant here,
// and the pads the model uses are multiples of 8 with power-of-two divisors in
// practice.

/// Real 2-D FFT with `norm='ortho'`; writes `[2][h][w/2+1]` (real then
/// imaginary) exactly like `stack([rfft.real, rfft.imag])`.
pub fn rfft2_ortho(src: &[f32], h: usize, w: usize, dst: &mut [f32]) {
    let hw = w / 2 + 1;
    let (mut re, mut im) = (vec![0f64; h * w], vec![0f64; h * w]);
    for (i, v) in src.iter().enumerate() {
        re[i] = *v as f64;
    }
    fft2(&mut re, &mut im, h, w, false);
    let scale = 1.0 / ((h * w) as f64).sqrt();
    for y in 0..h {
        for x in 0..hw {
            dst[y * hw + x] = (re[y * w + x] * scale) as f32;
            dst[h * hw + y * hw + x] = (im[y * w + x] * scale) as f32;
        }
    }
}

/// Inverse of `rfft2_ortho`.
///
/// `torch.fft.irfftn(x, s=(h, w), norm='ortho')` treats its input as the
/// `[0..hw)` half-plane of a 2-D Hermitian spectrum and rebuilds the full plane
/// by reflection through the origin in BOTH axes:
///
/// ```text
///   F[y, x]           = X[y, x]
///   F[(h-y) % h, (w-x) % w] = conj(X[y, x])
/// ```
///
/// Filling only the `x` axis (conjugating each row independently) coincides with
/// this whenever the spectrum is Hermitian in `y` as well - which it is after
/// `rfft2_ortho` but emphatically is not after the 1x1 convolution inside
/// `FourierUnit`, whose output feeds this function.  The two constructions
/// differ by 0.306 on real data if only `x` is filled.
pub fn irfft2_ortho(src: &[f32], h: usize, w: usize, dst: &mut [f32]) {
    let hw = w / 2 + 1;
    // The half-plane is read from `src` throughout, never from the output, so a
    // mirror that lands back inside the stored columns (`w - x < hw`, which
    // covers the Nyquist column of an even `w`) cannot clobber a value that is
    // still to be read.
    let (mut re, mut im) = (vec![0f64; h * w], vec![0f64; h * w]);
    // Copy the stored half-plane, then fill the missing columns from it.  Reads
    // always come from `src` and writes only ever touch columns `>= hw`, so no
    // assignment can clobber a value that is still to be read - which matters
    // because the reflection maps some stored columns onto other stored columns
    // (for even `w` the Nyquist column is its own mirror).
    for y in 0..h {
        for x in 0..hw {
            re[y * w + x] = src[y * hw + x] as f64;
            im[y * w + x] = src[h * hw + y * hw + x] as f64;
        }
    }
    for y in 0..h {
        for x in hw..w {
            let xm = w - x;
            // The partner column of the Nyquist column is itself, and it is
            // already stored; nothing to fill.
            if xm >= hw {
                continue;
            }
            let ym = (h - y) % h;
            re[y * w + x] = re[ym * w + xm];
            im[y * w + x] = -im[ym * w + xm];
        }
    }
    fft2(&mut re, &mut im, h, w, true);
    let scale = 1.0 / ((h * w) as f64).sqrt();
    for i in 0..h * w {
        dst[i] = (re[i] * scale) as f32;
    }
}

/// 2-D complex FFT of an `h x w` row-major plane: rows first, then columns.
///
/// Rows and columns are independent transforms of small vectors, so the work is
/// spread across threads.  Each row's pair of scratch vectors is private to the
/// thread that owns that row; the column pass needs the transposed access, which
/// is done by gathering into a private buffer rather than by transposing the
/// whole plane.
fn fft2(re: &mut [f64], im: &mut [f64], h: usize, w: usize, inverse: bool) {
    // Sequential on purpose: the caller parallelises over channels, and one
    // rayon dispatch per row/column of a 64x64 plane costs more than the
    // transform itself.
    let mut rr = vec![0f64; w];
    let mut ii = vec![0f64; w];
    for y in 0..h {
        rr.copy_from_slice(&re[y * w..(y + 1) * w]);
        ii.copy_from_slice(&im[y * w..(y + 1) * w]);
        fft(&mut rr, &mut ii, inverse);
        re[y * w..(y + 1) * w].copy_from_slice(&rr);
        im[y * w..(y + 1) * w].copy_from_slice(&ii);
    }
    let (mut cr, mut ci) = (vec![0f64; h], vec![0f64; h]);
    for x in 0..w {
        for y in 0..h {
            cr[y] = re[y * w + x];
            ci[y] = im[y * w + x];
        }
        fft(&mut cr, &mut ci, inverse);
        for y in 0..h {
            re[y * w + x] = cr[y];
            im[y * w + x] = ci[y];
        }
    }
}

/// Complex 1-D FFT of arbitrary length: radix-2 when `n` is a power of two,
/// Bluestein's chirp-z otherwise.  Unnormalised (the callers apply `norm`).
fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        fft_radix2(re, im, inverse);
    } else {
        fft_bluestein(re, im, inverse);
    }
}

/// Iterative radix-2 Cooley-Tukey FFT.  `n` must be a power of two.
fn fft_radix2(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let ang = sign * 2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let (mut cr, mut ci) = (1.0f64, 0.0f64);
            for k in 0..len / 2 {
                let (ur, ui) = (re[i + k], im[i + k]);
                let vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
                let vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Bluestein's tables for one transform length: the chirp, its wrapped
/// spectrum (`fft(b)`) and the transform size.
struct BluesteinTables {
    m: usize,
    wr: Vec<f64>,
    wi: Vec<f64>,
    br: Vec<f64>,
    bi: Vec<f64>,
}

thread_local! {
    /// Per-thread cache keyed by `(n, inverse)`.  Planes repeat the same two or
    /// three lengths for every one of the ~200 channels, so rebuilding the chirp
    /// and re-transforming `b` on each call dominated the FFT.
    static BLUESTEIN: std::cell::RefCell<std::collections::HashMap<(usize, bool), std::rc::Rc<BluesteinTables>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

fn bluestein_tables(n: usize, inverse: bool) -> std::rc::Rc<BluesteinTables> {
    BLUESTEIN.with(|cell| {
        let mut map = cell.borrow_mut();
        if let Some(t) = map.get(&(n, inverse)) {
            return t.clone();
        }
        let sign = if inverse { 1.0 } else { -1.0 };
        // Chirp w_j = exp(sign * i * pi * j^2 / n); using j^2 mod 2n avoids the
        // precision loss of squaring large j.
        let mut wr = vec![0f64; n];
        let mut wi = vec![0f64; n];
        for j in 0..n {
            let jm = ((j as i64 * j as i64) % (2 * n as i64)) as f64;
            let ang = sign * std::f64::consts::PI * jm / n as f64;
            wr[j] = ang.cos();
            wi[j] = ang.sin();
        }
        let m = (2 * n - 1).next_power_of_two();
        // b_j = w_j, wrapped around both ends, transformed once and reused.
        let (mut br, mut bi) = (vec![0f64; m], vec![0f64; m]);
        br[0] = wr[0];
        bi[0] = wi[0];
        for j in 1..n {
            br[j] = wr[j];
            bi[j] = wi[j];
            br[m - j] = wr[j];
            bi[m - j] = wi[j];
        }
        fft_radix2(&mut br, &mut bi, false);
        let t = std::rc::Rc::new(BluesteinTables { m, wr, wi, br, bi });
        map.insert((n, inverse), t.clone());
        t
    })
}

/// Bluestein's algorithm: `X_k = conj(w_k) * DFT(a * b)_k` where the two
/// sequences are convolved with a power-of-two FFT.
fn fft_bluestein(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    let t = bluestein_tables(n, inverse);
    let BluesteinTables { m, ref wr, ref wi, ref br, ref bi } = *t;

    // a_j = x_j * conj(w_j)
    let (mut ar, mut ai) = (vec![0f64; m], vec![0f64; m]);
    for j in 0..n {
        ar[j] = re[j] * wr[j] + im[j] * wi[j];
        ai[j] = im[j] * wr[j] - re[j] * wi[j];
    }

    fft_radix2(&mut ar, &mut ai, false);
    for i in 0..m {
        let xr = ar[i] * br[i] - ai[i] * bi[i];
        let xi = ar[i] * bi[i] + ai[i] * br[i];
        ar[i] = xr;
        ai[i] = xi;
    }
    fft_radix2(&mut ar, &mut ai, true);
    let inv_m = 1.0 / m as f64;
    for j in 0..n {
        let cr = ar[j] * inv_m;
        let ci = ai[j] * inv_m;
        re[j] = cr * wr[j] + ci * wi[j];
        im[j] = ci * wr[j] - cr * wi[j];
    }
}
