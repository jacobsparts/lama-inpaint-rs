//! Compiles the two halves of the embedded kernel image: the shared `lightgpu`
//! toolkit's `cuda/kernels.cu` (TOOLKIT_KERNELS) and this engine's own
//! `cuda/lama.cu` (PROJECT_KERNELS).
//!
//! Each compiles to its own fatbin with its own `--entries` list and `src/cuda.rs`
//! loads them as separate modules, so neither can shadow a name in the other.
//! A kernel missing from its list is PRUNED from the fatbin and fails at launch
//! rather than at build time, so both lists are checked against the source they
//! are compiled from before nvcc runs.

/// Generic ops from the shared toolkit. big-lama calls the batched 2-D Fourier
/// transforms behind its spectral blocks (originally an in-tree implementation,
/// lifted into the toolkit when it turned out to be model-independent), plus
/// `lg_sigmoid` for the final output layer - this engine's own `k_sigmoid` was
/// the toolkit's kernel character for character, so it is gone. Everything else
/// here is specific to big-lama's architecture and lives in `cuda/lama.cu`.
const TOOLKIT_KERNELS: &[&str] = &[
    "lg_fft2_r2c",
    "lg_fft2_c2r",
    "lg_sigmoid",
    // Promoted from this engine's own file: an exact duplicate of an in-place
    // plane accumulate, and of a plane copy. See the note over PROJECT_KERNELS.
    "lg_add_inplace",
    "lg_copy",
];

/// This engine's own kernels, in `cuda/lama.cu`.
///
/// `k_add_inplace` and `k_copy_plane` are no longer here: they were exact
/// duplicates of the toolkit's `lg_add_inplace` and `lg_copy` and now live in
/// `TOOLKIT_KERNELS` above instead. `k_scale` is gone entirely - it was listed
/// here but never launched, so it only ever cost fatbin bytes.
const PROJECT_KERNELS: &[&str] = &[
    "k_im2col",
    "k_conv_transpose",
    "k_bn_relu",
    "k_reflect_pad",
    "k_avgpool2x2",
    "k_spec_pack",
    "k_spec_unpack",
    "k_convt_col",
    "k_convt_put",
    "k_bias_plane",
    "k_conv7x7",
    "k_bn_relu_add",
    "k_sgemm_slab",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/lama.cu");

    let toolkit = lightgpu_build::toolkit_kernels_cu()
        .expect("locate lightgpu's cuda/kernels.cu (set LA_GPU_DIR to override)");
    let toolkit = toolkit.to_string_lossy().into_owned();

    for k in TOOLKIT_KERNELS {
        assert!(
            lightgpu_build::known_kernel(k),
            "unknown kernel `{k}`: not defined by the toolkit - did it move into cuda/lama.cu?"
        );
    }
    let src = std::fs::read_to_string("cuda/lama.cu").expect("read cuda/lama.cu");
    let defined = lightgpu_build::kernel_names_in(&src);
    for k in PROJECT_KERNELS {
        assert!(
            defined.iter().any(|d| d == k),
            "`{k}` is not defined in cuda/lama.cu (it has {})",
            defined.join(", ")
        );
    }

    lightgpu_build::fatbin_modules(&[
        lightgpu_build::Source {
            path: &toolkit,
            out_name: "lama_toolkit.fatbin",
            entries: Some(TOOLKIT_KERNELS),
        },
        lightgpu_build::Source {
            path: "cuda/lama.cu",
            out_name: "lama_project.fatbin",
            entries: Some(PROJECT_KERNELS),
        },
    ]);
}
