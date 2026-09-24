//! Weights: the big-lama reading of a `.safetensors` checkpoint.
//!
//! The container itself - `mmap`, the JSON header, tensor offsets and borrowed
//! slices - is `lightgpu::safetensors`, shared with every other engine in the
//! family.  What is left here is the part that is specific to this engine:
//!
//! * `f32(name)` copies a tensor out as host `f32`, which is what the CPU path
//!   and the tensors the GPU loader *computes* (BN folds, FFC modulations) need;
//! * the GPU path uploads the whole mapping in one `cuMemcpyHtoD` and addresses
//!   each verbatim tensor at `arena + info.offset`, so it needs the raw bytes
//!   ([`WeightStore::bytes`]) and absolute offsets ([`WeightStore::info`]);
//! * every tensor is checked against its `data_offsets` before anything is
//!   handed out, so a truncated or mis-converted file fails at load.
//!
//! The order in which tensors appear in the container is the layer order the
//! network walks; `export_weights.py` writes them in `state_dict` order for
//! exactly that reason.
//!
//! Note the container is a real safetensors file with a JSON header, and a
//! 204 MB header would be silly, so the reader keeps the header (about 100 KB)
//! parsed in memory and borrows only the payloads.  A caller that uploads the
//! whole mapping to the GPU copies the header too; it is a rounding error
//! against 204 MB.

use std::path::Path;

use lightgpu::safetensors;

/// One tensor in the container, as this engine's code has always seen it.
#[allow(dead_code)]  // `name` is kept for diagnostics/error messages.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    /// Absolute byte offset in the file, so an arena that is a copy of the
    /// whole file can address the tensor at `arena + offset`.
    pub offset: usize,
    pub nbytes: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

impl From<safetensors::TensorInfo> for TensorInfo {
    fn from(t: safetensors::TensorInfo) -> TensorInfo {
        TensorInfo { name: t.name, shape: t.shape, offset: t.offset, nbytes: t.nbytes }
    }
}

impl TensorInfo {
    fn from_info(t: &safetensors::TensorInfo) -> TensorInfo {
        TensorInfo {
            name: t.name.clone(),
            shape: t.shape.clone(),
            offset: t.offset,
            nbytes: t.nbytes,
        }
    }
}

pub struct WeightIndex {
    pub tensors: std::collections::HashMap<String, TensorInfo>,
    pub order: Vec<String>,
}

/// The checkpoint: `lightgpu::safetensors`' mapped file plus this engine's
/// integer index of it.
pub struct WeightStore {
    pub file: safetensors::File,
    pub index: WeightIndex,
}

impl WeightStore {
    /// Open `path`.  `map` mmaps the file (the GPU path wants that: the mapping
    /// goes to the device in one copy and the pages stay clean); without it the
    /// file is read into the heap.
    pub fn open(path: &Path, map: bool) -> Result<Self, String> {
        let file = if map {
            safetensors::File::open(path)
        } else {
            safetensors::File::read(path)
        }?;

        // Every tensor must be FP32; the kernels assume it and a quantized file
        // should fail here rather than be read as if it were not.
        let mut tensors = std::collections::HashMap::new();
        let mut order = Vec::new();
        for name in file.order() {
            let info = file.info(name)?;
            if info.dtype != safetensors::DType::F32 {
                return Err(format!(
                    "tensor {name} is {:?}, not F32 - is this a big-lama checkpoint?",
                    info.dtype
                ));
            }
            order.push(name.clone());
            tensors.insert(name.clone(), TensorInfo::from_info(info));
        }
        if order.is_empty() {
            return Err(format!("no tensors found in {}", path.display()));
        }
        Ok(Self { file, index: WeightIndex { tensors, order } })
    }

    pub fn get(&self, name: &str) -> Result<&TensorInfo, String> {
        self.index
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor in weight blob: {name}"))
    }

    /// Look up a tensor's shape/offset without copying its data.
    pub fn info(&self, name: &str) -> Result<&TensorInfo, String> {
        self.get(name)
    }

    /// The whole container, so the GPU path can upload it in one `cuMemcpyHtoD`
    /// and address each tensor at `arena + info.offset`.
    pub fn bytes(&self) -> &[u8] {
        self.file.as_slice()
    }

    pub fn total_bytes(&self) -> usize {
        self.index
            .order
            .iter()
            .filter_map(|n| self.get(n).ok())
            .map(|t| t.nbytes)
            .sum()
    }

    /// Copy a tensor out of the container as host `f32`.
    pub fn f32(&self, name: &str) -> Result<Vec<f32>, String> {
        Ok(self.file.f32(name)?.to_vec())
    }
}
