
// big-lama's own device kernels, plus the container the fatbin is built into.
//
// This file is the PROJECT half of the kernel image. The generic half - the
// batched 2-D Fourier transforms this engine's spectral blocks call - lives in
// the shared `lightgpu` toolkit and is compiled as a SEPARATE module by
// build.rs, which loads both. Keeping them apart is what makes the toolkit's
// kernels reusable: each file compiles with its own `--entries` list, so neither
// can shadow the other's names.
//
// Compiled by build.rs (through lightgpu-build) and embedded as a fatbin, so the
// Rust build needs nvcc but neither the CUDA headers on the include path nor a
// driver-side JIT step.
//
// Conventions match the CPU engine in cpu.rs and PyTorch: planes are row-major
// [h][w] inside a contiguous [c][h][w] buffer.

// im2col for a kxk convolution.  `col` is [cin*kh*kw][oh*ow] with the patch index
// outermost, which is the layout `k_sgemm_slab` consumes as A.
extern "C" __global__ void k_im2col(
    const float* __restrict__ src, float* __restrict__ col,
    int cin, int h, int w, int kh, int kw,
    int pad_h, int pad_w, int stride, int oh, int ow, int reflect)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)cin * kh * kw * oh * ow;
    if (idx >= total) return;
    int ox = (int)(idx % ow);
    long long t = idx / ow;
    int oy = (int)(t % oh); t /= oh;
    int kx = (int)(t % kw); t /= kw;
    int ky = (int)(t % kh); t /= kh;
    int ci = (int)t;
    int iy = oy * stride + ky - pad_h;
    int ix = ox * stride + kx - pad_w;
    float v = 0.f;
    if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
        v = src[((long long)ci * h + iy) * w + ix];
    } else if (reflect) {
        // PyTorch reflect padding: mirror about the edge, edge sample not repeated.
        if (h > 1) { int p = 2 * (h - 1); iy = ((iy % p) + p) % p; if (iy >= h) iy = p - iy; }
        else iy = 0;
        if (w > 1) { int p = 2 * (w - 1); ix = ((ix % p) + p) % p; if (ix >= w) ix = p - ix; }
        else ix = 0;
        v = src[((long long)ci * h + iy) * w + ix];
    }
    col[idx] = v;
}

// Transposed 3x3 stride-2 convolution (the learned upsample).  Gathered rather
// than scattered so no atomics and no zero-fill are needed.
// weight layout is [cin][cout][kh][kw]; output is [cout][oh][ow].
extern "C" __global__ void k_conv_transpose(
    const float* __restrict__ in, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ out,
    int cin, int cout, int h, int w, int oh, int ow, int kh, int kw, int stride, int pad)
{
    // stride 2, pad 1, k 3: only taps whose (oy + pad - ky) is even and lands in
    // range contribute, so the valid ky set depends only on `oy % stride`.  On
    // the grid-stride loop each thread walks several outputs, so the tap sets are
    // resolved once per step instead of per tap.
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)cout * oh * ow;
    long long step = (long long)gridDim.x * blockDim.x;
    for (; idx < total; idx += step) {
        int ox = (int)(idx % ow);
        long long t = idx / ow;
        int oy = (int)(t % oh);
        int co = (int)(t / oh);
        float acc = bias ? bias[co] : 0.f;
        // Valid ky for this row: ky == (oy + pad) mod stride, then + stride.
        int y0 = ((oy + pad) % stride + stride) % stride;
        for (int ky = y0; ky < kh; ky += stride) {
            int iy = (oy + pad - ky) / stride;
            if (iy < 0 || iy >= h) continue;
            int x0 = ((ox + pad) % stride + stride) % stride;
            for (int kx = x0; kx < kw; kx += stride) {
                int ix = (ox + pad - kx) / stride;
                if (ix < 0 || ix >= w) continue;
                const float* kp = weight + ((long long)co * kh + ky) * kw + kx;
                // weight is [cin][cout][kh][kw]; stride over ci.
                for (int ci = 0; ci < cin; ++ci) {
                    acc += in[(long long)ci * h * w + (long long)iy * w + ix] * kp[(long long)ci * cout * kh * kw];
                }
            }
        }
        out[idx] = acc;
    }
}

// BatchNorm (folded to scale/shift) followed by ReLU, per channel.
extern "C" __global__ void k_bn_relu(float* __restrict__ x, const float* __restrict__ scale,
                          const float* __restrict__ shift, long long plane, int channels)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = plane * channels;
    if (idx >= total) return;
    int c = (int)(idx / plane);
    float v = x[idx] * scale[c] + shift[c];
    x[idx] = v > 0.f ? v : 0.f;
}

// `k_add_inplace` and `k_copy_plane` used to be here. Both are the toolkit's: an
// in-place accumulate over a plane is `lg_add_inplace` (identical body), and a
// plain copy is `lg_copy` (identical body under different pointer names). A
// duplicate of a shared kernel is not an optimisation waiting to happen, it is a
// second definition of one operation - the thing the shared set exists to avoid.
//
// The one thing to check when moving a call site across is the LENGTH WIDTH:
// these two took a `long long`, where the toolkit's take `int` and `long`. The
// kernels read the length from the argument slot it sits in, so passing it at the
// wrong width is a wrong number rather than an error - see CONVENTIONS.md section
// 1 in the toolkit.
//
// `k_scale` was also here and has been deleted rather than promoted: it had no
// call site at all, and it was not the toolkit's `lg_scale` anyway (that one is
// out-of-place, `y = x * s`, indexed by a separate output pointer; this one was
// in-place `x *= s` with the scalar after the length). If a scale is needed, the
// toolkit op is the one to call - or `lg_channel_affine`'s null-shift case, which
// is in-place and per-channel.

// ReflectionPad2d over [c][h][w] -> [c][h+2p][w+2p].
extern "C" __global__ void k_reflect_pad(const float* __restrict__ src, float* __restrict__ dst,
                              int c, int h, int w, int pad)
{
    int oh = h + 2 * pad, ow = w + 2 * pad;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)c * oh * ow;
    if (idx >= total) return;
    int x = (int)(idx % ow);
    long long t = idx / ow;
    int y = (int)(t % oh);
    int ch = (int)(t / oh);
    int sy = y - pad, sx = x - pad;
    if (h > 1) { int p = 2 * (h - 1); sy = ((sy % p) + p) % p; if (sy >= h) sy = p - sy; } else sy = 0;
    if (w > 1) { int p = 2 * (w - 1); sx = ((sx % p) + p) % p; if (sx >= w) sx = p - sx; } else sx = 0;
    dst[idx] = src[((long long)ch * h + sy) * w + sx];
}

// 2x2 stride-2 average pool (the SpectralTransform downsample for stride 2).
extern "C" __global__ void k_avgpool2x2(const float* __restrict__ src, float* __restrict__ dst,
                             int c, int h, int w)
{
    int oh = h / 2, ow = w / 2;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)c * oh * ow;
    if (idx >= total) return;
    int x = (int)(idx % ow);
    long long t = idx / ow;
    int y = (int)(t % oh);
    int ch = (int)(t / oh);
    const float* p = src + (long long)ch * h * w;
    float a = p[(long long)(2 * y) * w + 2 * x];
    float b = p[(long long)(2 * y) * w + 2 * x + 1];
    float cc = p[(long long)(2 * y + 1) * w + 2 * x];
    float d = p[(long long)(2 * y + 1) * w + 2 * x + 1];
    dst[idx] = (a + b + cc + d) * 0.25f;
}

// The R2C transform writes an interleaved complex plane per channel; the
// spectral 1x1 convolution expects the reference's channel-major stacking, i.e.
// for each channel a contiguous real plane followed by a contiguous imaginary
// plane (c0.re, c0.im, c1.re, c1.im, ...).  `k_spec_pack` produces that layout
// and `k_spec_unpack` reverses it before the inverse transform.
// Pack the interleaved complex spectrum into the stacked real/imag channel
// layout the 1x1 convolution consumes, applying the 1/sqrt(h*w) ortho factor of
// the forward transform in the same pass (the transform is unnormalised).
// Folding the scale in here saves a whole extra kernel launch and a pass over
// the data per spectral call.
extern "C" __global__ void k_spec_pack(const float* __restrict__ cplx, float* __restrict__ out,
                            int half, int plane, float scale)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)half * plane) return;
    long long c = idx / plane;
    long long i = idx - c * plane;
    out[c * 2 * plane + i] = cplx[2 * idx] * scale;
    out[c * 2 * plane + plane + i] = cplx[2 * idx + 1] * scale;
}

extern "C" __global__ void k_spec_unpack(const float* __restrict__ in, float* __restrict__ cplx, int half, int plane)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)half * plane) return;
    long long c = idx / plane;
    long long i = idx - c * plane;
    cplx[2 * idx] = in[c * 2 * plane + i];
    cplx[2 * idx + 1] = in[c * 2 * plane + plane + i];
}

// Transposed convolution as four phase GEMMs.  With stride 2, pad 1 and a 3x3
// kernel an output pixel (oy, ox) only sees the taps whose (oy + 1 - ky) and
// (ox + 1 - kx) are even, so the tap set depends only on the phase
// (oy % 2, ox % 2): 1 tap for (0,0), 2 for (0,1) and (1,0), 4 for (1,1).
//
// `k_convt_col` gathers, for one phase, the patch matrix [K][n] (K = cin*taps,
// n = number of outputs in the phase) laid out exactly like im2col so
// `k_sgemm_slab` can consume it.  The phase tap order is row-major over (ky, kx) restricted to
// the phase's tap set, matching the weight reorder done at load time in cuda.rs.
// The tap (ky, kx) pairs for each output phase (oy%2, ox%2), in the order the
// host reorders the weights into.  With stride 2, pad 1 and k = 3 only taps
// with even (oy + 1 - ky) and (ox + 1 - kx) contribute.
__device__ const int CONVT_KY[4][4] = {{1, 0, 0, 0}, {1, 1, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 2}};
__device__ const int CONVT_KX[4][4] = {{1, 0, 0, 0}, {0, 2, 0, 0}, {1, 1, 0, 0}, {0, 2, 0, 2}};

extern "C" __global__ void k_convt_col(
    const float* __restrict__ in, float* __restrict__ col,
    int cin, int h, int w, int py, int px, int taps, int n, int oh, int ow)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)cin * taps * n) return;
    int j = (int)(idx % n);
    long long t = idx / n;
    int ti = (int)(t % taps);
    int ci = (int)(t / taps);
    // The phase grid holds every other output pixel, so its width is half the
    // output width (rounded up for the odd phase).
    int gw = (ow + 1 - px) / 2;
    int i = j / gw;   // phase row index
    int jj = j % gw;  // phase column index
    int oy = 2 * i + py;
    int ox = 2 * jj + px;
    int ph = py * 2 + px;
    int ky = CONVT_KY[ph][ti];
    int kx = CONVT_KX[ph][ti];
    int iy = (oy + 1 - ky) / 2;
    int ix = (ox + 1 - kx) / 2;
    float v = 0.f;
    if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
        v = in[((long long)ci * h + iy) * w + ix];
    }
    col[((long long)ci * taps + ti) * n + j] = v;
}

extern "C" __global__ void k_convt_put(
    const float* __restrict__ out_phase, float* __restrict__ out,
    const float* __restrict__ bias, int cout, int n, int py, int px, int oh, int ow,
    int add_bias)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)cout * n) return;
    int j = (int)(idx % n);
    long long co = idx / n;
    int gw = (ow + 1 - px) / 2;
    int i = j / gw;
    int jj = j % gw;
    int oy = 2 * i + py;
    int ox = 2 * jj + px;
    float v = out_phase[idx];
    if (add_bias && bias) v += bias[co];
    out[(co * oh + oy) * ow + ox] = v;
}

// Add a per-channel bias to a `[cout][n]` plane in place.  Used by the
// convolution paths, which own their output buffers; doing it on the device
// avoids materialising a `cout*n` broadcast plane on the host.
extern "C" __global__ void k_bias_plane(float* __restrict__ out, const float* __restrict__ bias,
                             int cout, long long n)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)cout * n) return;
    out[idx] += bias[idx / n];
}

// Direct 7x7 convolution, one thread per output spatial position.
//
// The OutConv is 7x7 with only three output channels over a 518x518 plane, so
// the im2col path materialises a `(cin*49) x (h*w)` patch matrix - 3.37 GB of
// traffic at 512x512 - to feed a 5 GFLOP GEMM, only 1.5 FLOP/byte.  Here each
// thread walks its 7x7 receptive field directly and keeps the output
// accumulators in registers: no patch matrix, and the overlapping reads of
// neighbouring threads hit cache.  The caller has already applied the
// reflection padding, so plain bounds checks are enough.
//
// COUT_MAX caps the register accumulators; the host picks this kernel only when
// `cout <= COUT_MAX` and otherwise falls back to im2col + the slab GEMM.
#define COUT_MAX 8

extern "C" __global__ void k_conv7x7(const float* __restrict__ in, const float* __restrict__ w,
                          const float* __restrict__ bias, float* __restrict__ out,
                          int cin, int cout, int h_in, int w_in, int h_out, int w_out)
{
    // Output geometry drives the indexing; the input has its own (larger) extent
    // and row stride - conflating the two silently reads the wrong rows whenever
    // the input and output widths differ, which they do here (518 -> 512).
    long long n = (long long)h_out * w_out;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int y = (int)(idx / w_out);
    int x = (int)(idx % w_out);
    float acc[COUT_MAX];
    for (int co = 0; co < COUT_MAX; ++co) acc[co] = 0.f;
    // Slide the kernel over the receptive field.  The inner loop walks x with a
    // fixed dy, so the input reads are contiguous and the weight reads are the
    // small 7x7xcin block reused by every thread.
    long long in_plane = (long long)h_in * w_in;
    for (int ci = 0; ci < cin; ++ci) {
        const float* plane = in + (long long)ci * in_plane;
        const float* kp = w + ((long long)ci * cout) * 49;
        for (int dy = 0; dy < 7; ++dy) {
            int iy = y + dy;
            if (iy >= h_in) break;
            const float* row = plane + (long long)iy * w_in;
            for (int dx = 0; dx < 7; ++dx) {
                int ix = x + dx;
                if (ix >= w_in) break;
                float v = row[ix];
                // `kp` is [cout][7][7] for this input channel; each output
                // channel reads its own 49-element slice.
                const float* wc = kp + (long long)dy * 7 + dx;
                for (int co = 0; co < cout && co < COUT_MAX; ++co) {
                    acc[co] += v * wc[(long long)co * 49];
                }
            }
        }
    }
    for (int co = 0; co < cout && co < COUT_MAX; ++co) {
        float b = bias ? bias[co] : 0.f;
        out[(long long)co * n + idx] = acc[co] + b;
    }
}


// Fused BatchNorm + ReLU + residual add: `out = relu(scale*x + shift) + skip`.
//
// The resblock tail otherwise needs two passes over the activation (a BN+ReLU
// and an add) plus a full device-to-device copy of the input for the skip.  One
// pass keeps the activation hot in L2 and removes a launch and a copy.
extern "C" __global__ void k_bn_relu_add(float* __restrict__ x, const float* __restrict__ skip,
                              const float* __restrict__ scale, const float* __restrict__ shift,
                              long long plane, int channels)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= plane * channels) return;
    int ch = (int)(idx / plane);
    float v = x[idx] * scale[ch] + shift[ch];
    v = v > 0.f ? v : 0.f;
    x[idx] = skip ? v + skip[idx] : v;
}


// Single-precision GEMM for every convolution path.
//
//   C[o][j] = sum_k B[o][k] * A[k][j]
//
// with A (k x n) row-major ld = lda, B (cout x k) row-major ld = ldb, C (cout x n)
// row-major ld = ldc, so no call site needs transpose work.
// Requirements: k >= 4 and k % 4 == 0 (k is 9*cin,
// 4*cin or cin*taps and cin is always a multiple of 64); n and cout may be anything,
// since the tiles are predicated and the C store falls back to scalars on the edge.
//
// Shape: 256 threads own a 64-column x 64-row tile of C; thread (tx = tid & 15,
// ty = tid >> 4) owns columns c0 + tx*4 .. +3 of rows r0 + ty*4 .. +3, sixteen
// accumulators.  The k loop walks a 16-wide slab at a time: the block stages A's 16
// k-rows (16x68 floats) AND B's 64 rows (64x17 floats) into shared memory, then runs
// four k-steps reading both operands from shared.
//
// Staging B as well as A is what makes this fast: with 16 column lanes sharing a row
// group, reloading B from global memory per k-step would fetch the same values 16
// times per block per step.  `b_align` is unused here - the staging reads B one scalar
// per thread, which is alignment independent - but it stays in the signature so the
// launcher marshals the same ten arguments for every kernel.
#define KS 16
extern "C" __global__ void k_sgemm_slab(const float* __restrict__ a, const float* __restrict__ b,
                       float* __restrict__ c, int lda, int ldb, int ldc,
                       int k, int cout, int n, int nb_cols, int b_align)
{
    __shared__ float as[KS][68];
    __shared__ float bs[64][KS + 1];
    const int tx = threadIdx.x & 15;
    const int ty = threadIdx.x >> 4;
    const int c0 = (blockIdx.x % nb_cols) * 64;
    const int r0 = (blockIdx.x / nb_cols) * 64;
    const int cA = c0 + tx * 4;
    const int o0 = r0 + ty * 4;
    (void)b_align;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < k; k0 += KS) {
        // Stage A: KS rows x 64 columns.  256 threads cover 16 rows of 64 in 4 passes.
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int r = p * 4 + (threadIdx.x >> 6);
            const int cc = c0 + (threadIdx.x & 63);
            const int kk = k0 + r;
            as[r][threadIdx.x & 63] = (kk < k && cc < n) ? a[(long long)kk * lda + cc] : 0.f;
        }
        // Stage B: 64 rows x KS k-values.  256 threads cover 64 rows x 4 columns per pass.
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int row = p * 16 + (threadIdx.x >> 4);
            const int q = threadIdx.x & 15;
            const int oo = r0 + row;
            const int kk = k0 + q;
            float v = 0.f;
            if (oo < cout && kk < k) {
                v = b[(long long)oo * ldb + kk];
            }
            bs[row][q] = v;
        }
        __syncthreads();
        #pragma unroll
        for (int ks = 0; ks + 4 <= KS; ks += 4) {
            if (k0 + ks >= k) break;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float bv4[4];
                if (o0 + i < cout) {
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) bv4[q] = bs[ty * 4 + i][ks + q];
                } else {
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) bv4[q] = 0.f;
                }
                const float4 av = *reinterpret_cast<const float4*>(&as[ks][tx * 4]);
                acc[i][0] = fmaf(bv4[0], av.x, acc[i][0]);
                acc[i][1] = fmaf(bv4[0], av.y, acc[i][1]);
                acc[i][2] = fmaf(bv4[0], av.z, acc[i][2]);
                acc[i][3] = fmaf(bv4[0], av.w, acc[i][3]);
                const float4 av1 = *reinterpret_cast<const float4*>(&as[ks + 1][tx * 4]);
                acc[i][0] = fmaf(bv4[1], av1.x, acc[i][0]);
                acc[i][1] = fmaf(bv4[1], av1.y, acc[i][1]);
                acc[i][2] = fmaf(bv4[1], av1.z, acc[i][2]);
                acc[i][3] = fmaf(bv4[1], av1.w, acc[i][3]);
                const float4 av2 = *reinterpret_cast<const float4*>(&as[ks + 2][tx * 4]);
                acc[i][0] = fmaf(bv4[2], av2.x, acc[i][0]);
                acc[i][1] = fmaf(bv4[2], av2.y, acc[i][1]);
                acc[i][2] = fmaf(bv4[2], av2.z, acc[i][2]);
                acc[i][3] = fmaf(bv4[2], av2.w, acc[i][3]);
                const float4 av3 = *reinterpret_cast<const float4*>(&as[ks + 3][tx * 4]);
                acc[i][0] = fmaf(bv4[3], av3.x, acc[i][0]);
                acc[i][1] = fmaf(bv4[3], av3.y, acc[i][1]);
                acc[i][2] = fmaf(bv4[3], av3.z, acc[i][2]);
                acc[i][3] = fmaf(bv4[3], av3.w, acc[i][3]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (o0 + i < cout) {
            if (cA + 3 < n) {
                float4 v;
                v.x = acc[i][0]; v.y = acc[i][1]; v.z = acc[i][2]; v.w = acc[i][3];
                *reinterpret_cast<float4*>(&c[(long long)(o0 + i) * ldc + cA]) = v;
            } else {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (cA + j < n) c[(long long)(o0 + i) * ldc + cA + j] = acc[i][j];
            }
        }
    }
}

