# lama-inpaint (Rust engine)

See the [repository README](../README.md) for the project overview, the
quick start, the weight files and the GPU requirements. This crate is the
standalone engine: `src/cuda.rs` (CUDA), `src/cpu.rs` (CPU), `src/model.rs`
(the plan) and `src/weights.rs` (the `.safetensors` weight store). Build with
`cargo build --release`; `build.rs` precompiles `cuda/lama.cu` into a fatbin
embedded in the binary, so nvcc is needed only to build, never to run.
