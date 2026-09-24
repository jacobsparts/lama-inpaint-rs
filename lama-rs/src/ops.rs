//! Numeric kernels shared by the GPU and CPU paths.
//!
//! The GPU path runs convolutions as its own slab SGEMM over an im2col patch
//! matrix and the Fourier units through the shared toolkit's batched 2-D
//! transforms (`lg_fft2_r2c`/`lg_fft2_c2r`, loaded as a second module - see
//! `cuda.rs`), launching everything through the driver API, so the binary runs
//! without CUDA headers, cuBLAS or cuFFT. Building the GPU path does need nvcc:
//! `build.rs` compiles `cuda/lama.cu` and the toolkit kernels into the embedded
//! image.
//! The CPU path is a direct convolution whose inner loops are cache-friendly
//! and rayon-parallel.

use rayon::prelude::*;

/// im2col: gather `(cin * kh * kw)` patches for a conv with the given padding.
///
/// Output layout is `[cin][kh][kw][oh][ow]` so the GEMM below reads it as a
/// column-major `(cin*kh*kw) x (oh*ow)` matrix with a contiguous inner axis.
pub fn im2col(
    src: &[f32],
    cin: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    pad_h: usize,
    pad_w: usize,
    oh: usize,
    ow: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; cin * kh * kw * oh * ow];
    let plane = oh * ow;
    for ci in 0..cin {
        let src_c = &src[ci * h * w..(ci + 1) * h * w];
        for ky in 0..kh {
            for kx in 0..kw {
                let dst = &mut out[((ci * kh + ky) * kw + kx) * plane..][..plane];
                dst.par_chunks_mut(ow).enumerate().for_each(|(oy, row)| {
                    let sy = oy as isize + ky as isize - pad_h as isize;
                    for (ox, v) in row.iter_mut().enumerate() {
                        let sx = ox as isize + kx as isize - pad_w as isize;
                        *v = if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                            src_c[sy as usize * w + sx as usize]
                        } else {
                            0.0
                        };
                    }
                });
            }
        }
    }
    out
}

/// Column-major `A (m x k) * B (k x n) -> C (m x n)` for the BLAS convention.
///
/// `patch` is the `[k][ow*oh]` im2col matrix (k = cin*kh*kw), `weights` is the
/// row-major `[cout][k]` filter matrix and `bias` is optional.
#[allow(clippy::too_many_arguments)]
pub fn gemm_nt_cpu(
    cout: usize,
    k: usize,
    n: usize,
    weights: &[f32],
    patch: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    out.par_chunks_mut(n).enumerate().for_each(|(co, row)| {
        let w = &weights[co * k..(co + 1) * k];
        let b = bias.map_or(0.0, |b| b[co]);
        for (o, v) in row.iter_mut().enumerate() {
            let mut acc = b;
            let pcol = &patch[o..];
            for (wi, p) in w.iter().zip(pcol.iter().step_by(1)) {
                acc += *wi * *p;
            }
            *v = acc;
        }
    });
}

/// Transposed convolution (learned upsample), CPU: for each input pixel scatter
/// its contribution into the output, which is cheaper than gathering when the
/// stride is 2 and the kernel is 3.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d_cpu(
    input: &[f32],
    cin: usize,
    h: usize,
    w: usize,
    weights: &[f32], // [cin][cout][kh][kw]
    cout: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    output_padding: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    let oh = (h - 1) * stride - 2 * pad + kh + output_padding;
    let ow = (w - 1) * stride - 2 * pad + kw + output_padding;
    for o in out.iter_mut() {
        *o = 0.0;
    }
    let out_plane = oh * ow;
    // Parallelise over input channels; each thread owns a disjoint slab of the
    // output, so no atomics are needed.
    input
        .par_chunks(plane_of(h, w))
        .enumerate()
        .for_each(|(ci, src)| {
            let mut local = vec![0f32; cout * out_plane];
            for co in 0..cout {
                let kern = &weights[(ci * cout + co) * kh * kw..][..kh * kw];
                let dst = &mut local[co * out_plane..(co + 1) * out_plane];
                for iy in 0..h {
                    for ix in 0..w {
                        let v = src[iy * w + ix];
                        if v == 0.0 {
                            continue;
                        }
                        for ky in 0..kh {
                            let oy = iy * stride + ky;
                            if oy < pad || oy - pad >= oh {
                                continue;
                            }
                            let oy = oy - pad;
                            for kx in 0..kw {
                                let ox = ix * stride + kx;
                                if ox < pad || ox - pad >= ow {
                                    continue;
                                }
                                dst[oy * ow + (ox - pad)] += v * kern[ky * kw + kx];
                            }
                        }
                    }
                }
            }
            // Fold this channel's contribution into the shared output buffer.
            // Done serially per chunk so the expensive inner loop above stays
            // free of synchronisation.
            // NOTE: callers guarantee `out` is already zeroed.
            add_in_place(out, &local, cout, out_plane);
        });
    if let Some(b) = bias {
        for co in 0..cout {
            let row = &mut out[co * out_plane..(co + 1) * out_plane];
            let bv = b[co];
            row.iter_mut().for_each(|v| *v += bv);
        }
    }
}

#[inline]
fn plane_of(h: usize, w: usize) -> usize {
    h * w
}

fn add_in_place(out: &mut [f32], local: &[f32], cout: usize, plane: usize) {
    for co in 0..cout {
        let d = &mut out[co * plane..(co + 1) * plane];
        let s = &local[co * plane..(co + 1) * plane];
        for (a, b) in d.iter_mut().zip(s.iter()) {
            *a += *b;
        }
    }
}
