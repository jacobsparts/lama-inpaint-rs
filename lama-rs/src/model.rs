//! big-lama generator: the network description, resolved against the weight
//! blob, plus the CPU forward.
//!
//! The description mirrors `lamacore.py` / the original `ffc.py` module tree
//! exactly so parameter names line up with the checkpoint:
//!
//! ```text
//!   model.0  ReflectionPad2d(3)
//!   model.1  FFC_BN_ACT(4 -> 64,     k7, pad 0)         local only
//!   model.2  FFC_BN_ACT(64 -> 128,   k3, s2, pad 1)     local only
//!   model.3  FFC_BN_ACT(128 -> 256,  k3, s2, pad 1)     local only
//!   model.4  FFC_BN_ACT(256 -> 512,  k3, s2, pad 1)     ratio_gout 0.75
//!   model.5..22  FFCResnetBlock(512) x18                ratio 0.75 / 0.75
//!   model.23 ConcatTupleLayer
//!   model.24 ConvTranspose2d(a->b,k3,s2,p1,op1) + BN + ReLU
//!   model.27 .. model.30
//!   model.33 ReflectionPad2d(3)
//!   model.34 Conv2d(64 -> 3, k7)
//!   model.35 Sigmoid
//! ```
//!
//! Two conventions matter for the port:
//!
//! * Every spatial convolution in an FFC block uses `padding_mode='reflect'`
//!   (`ReflectionPad2d`), while the downsampling convolutions pad with zeros.
//! * `FourierUnit` applies `rfftn(norm='ortho')`, stacks real/imag so the 1x1
//!   convolution sees `2 * C` channels, then `irfftn(s=x.shape[-2:],
//!   norm='ortho')` at the input plane size.



/// One optional convolution inside an FFC block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvRef {
    /// Not present (Identity).
    Absent,
    /// Present; the weight lives under `model.{i}.ffc.{name}`.
    Present { name: &'static str, cin: usize, cout: usize },
}

/// An `FFC_BN_ACT`: local and global input halves, each optionally convolved.
///
/// Some of the fields below describe the block rather than drive the engine
/// (they mirror the reference architecture and are read only by the CUDA engine
/// when the plan changes shape), so unused ones are allowed here.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct FfcBlock {
    /// `model.{index}` for the parameter names.
    pub index: usize,
    /// Sub-module index within the layer (resnet blocks have two).
    pub sub: usize,
    pub cin: usize,
    pub cout: usize,
    pub kernel: usize,
    pub stride: usize,
    /// True when the spatial convolutions pad with zeros (the strided
    /// downsamplers); false when they reflect-pad.  `model.1` is special: its
    /// 7x7 convolution has `padding=0` and leans on the `ReflectionPad2d(3)`
    /// that precedes it, so `pad` is 0 rather than `kernel / 2`.
    pub zero_pad: bool,
    /// Explicit padding for the spatial convolutions.  Derived from the
    /// reference (`padding=dilation` for the resblocks, 0 for `model.1`).
    pub pad: usize,
    pub reflection: bool,
    pub in_local: usize,
    pub in_global: usize,
    pub out_local: usize,
    pub out_global: usize,
    pub l2l: ConvRef,
    pub l2g: ConvRef,
    pub g2l: ConvRef,
    /// Present when `out_global > 0`: the SpectralTransform parameters.
    pub g2g: bool,
    /// BatchNorm + ReLU on the local output.
    pub bn_local: bool,
    /// BatchNorm + ReLU on the global output.
    pub bn_global: bool,
}

/// A residual FFC block: two `FFC_BN_ACT`s plus a skip connection.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ResBlock {
    pub index: usize,
    pub conv1: FfcBlock,
    pub conv2: FfcBlock,
    /// Global channel count entering the block (for the inline split).
    pub in_global: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Upsample {
    pub index: usize,
    pub cin: usize,
    pub cout: usize,
    pub kernel: usize,
    pub stride: usize,
    pub pad: usize,
    pub output_padding: usize,
    pub bn_index: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct OutConv {
    pub index: usize,
    pub cin: usize,
    pub cout: usize,
    pub kernel: usize,
}

/// Ordered execution plan; the CPU forward walks it once, the GPU path
/// preallocates device buffers from the same description.
#[derive(Debug, Clone)]
pub enum Step {
    ReflectPad(usize),
    Ffc(FfcBlock),
    ResBlock(ResBlock),
    Concat,
    Upsample(Upsample),
    OutConv(OutConv),
}

#[derive(Debug, Clone)]
pub struct Model {
    pub steps: Vec<Step>,
}

const FFC_RESNET_RATIO: f32 = 0.75;

/// Build the plan. `cuda` only changes how the weight store materialises
/// tensors, not the arithmetic.
pub fn build() -> Model {
    let mut steps = Vec::new();
    steps.push(Step::ReflectPad(3));

    // model.1: 4 -> 64, k7, padding 0: the preceding ReflectionPad2d(3)
    // supplies the border, so the 7x7 convolution is "valid" and the plane
    // stays the same size.
    let mut init = ffc(1, 0, 4, 64, 7, 1, false, true, 0.0, 0.0);
    init.pad = 0;
    steps.push(Step::Ffc(init));

    // model.2, model.3: 64->128, 128->256; local only, stride 2.  `FFC`'s
    // default `padding_type` is "reflect" and it is forwarded as the
    // convolution's `padding_mode`, so the strided downsamplers reflect-pad
    // too - verified against torch (a zero pre-pad differs by ~1.4 on random
    // input, a reflect pre-pad is exact).
    steps.push(Step::Ffc(ffc(2, 0, 64, 128, 3, 2, false, true, 0.0, 0.0)));
    steps.push(Step::Ffc(ffc(3, 0, 128, 256, 3, 2, false, true, 0.0, 0.0)));
    // model.4: 256 -> 512, ratio_gout = 0.75 (global 384, local 128).
    steps.push(Step::Ffc(ffc(4, 0, 256, 512, 3, 2, false, true, 0.0, FFC_RESNET_RATIO)));

    // model.5..model.22: 18 calls to FFCResnetBlock, each with conv1/conv2.
    for index in 5..=22 {
        // sub 1 / 2 select `model.{i}.conv1` / `model.{i}.conv2` in the
        // parameter names; sub 0 is the bare `model.{i}.ffc` of a standalone
        // FFC_BN_ACT.
        let conv1 = ffc(index, 1, 512, 512, 3, 1, false, true, FFC_RESNET_RATIO, FFC_RESNET_RATIO);
        let conv2 = ffc(index, 2, 512, 512, 3, 1, false, true, FFC_RESNET_RATIO, FFC_RESNET_RATIO);
        let in_global = conv1.in_global;
        steps.push(Step::ResBlock(ResBlock { index, conv1, conv2, in_global }));
    }

    steps.push(Step::Concat);

    // model.24 / 27 / 30: ConvTranspose2d + BatchNorm2d + ReLU.
    // Channel counts account for the FFC split: after `ConcatTupleLayer` the
    // bottleneck carries local + global = 128 + 384 = 512 channels.
    steps.push(Step::Upsample(Upsample {
        index: 24,
        cin: 512,
        cout: 256,
        kernel: 3,
        stride: 2,
        pad: 1,
        output_padding: 1,
        bn_index: 25,
    }));
    steps.push(Step::Upsample(Upsample {
        index: 27,
        cin: 256,
        cout: 128,
        kernel: 3,
        stride: 2,
        pad: 1,
        output_padding: 1,
        bn_index: 28,
    }));
    steps.push(Step::Upsample(Upsample {
        index: 30,
        cin: 128,
        cout: 64,
        kernel: 3,
        stride: 2,
        pad: 1,
        output_padding: 1,
        bn_index: 31,
    }));

    steps.push(Step::ReflectPad(3));
    steps.push(Step::OutConv(OutConv { index: 34, cin: 64, cout: 3, kernel: 7 }));

    Model { steps }
}

/// Derive one FFC block's branch configuration from the reference rules.
#[allow(clippy::too_many_arguments)]
fn ffc(
    index: usize,
    sub: usize,
    cin: usize,
    cout: usize,
    kernel: usize,
    stride: usize,
    zero_pad: bool,
    reflection: bool,
    ratio_gin: f32,
    ratio_gout: f32,
) -> FfcBlock {
    let in_global = ((cin as f32) * ratio_gin) as usize;
    let out_global = ((cout as f32) * ratio_gout) as usize;
    let in_local = cin - in_global;
    let out_local = cout - out_global;

    let present = |a: usize, b: usize, name: &'static str| -> ConvRef {
        if a == 0 || b == 0 {
            ConvRef::Absent
        } else {
            ConvRef::Present { name, cin: a, cout: b }
        }
    };

    FfcBlock {
        index,
        sub,
        cin,
        cout,
        kernel,
        stride,
        zero_pad,
        pad: kernel / 2,
        reflection,
        in_local,
        in_global,
        out_local,
        out_global,
        l2l: present(in_local, out_local, "convl2l"),
        l2g: present(in_local, out_global, "convl2g"),
        g2l: present(in_global, out_local, "convg2l"),
        g2g: in_global > 0 && out_global > 0,
        bn_local: out_local > 0,
        bn_global: out_global > 0,
    }
}

/// Size of an FFC block's BatchNorm buffers, for logging.
pub fn describe(model: &Model) -> String {
    let mut ffc_count = 0;
    let mut res_blocks = 0;
    for s in &model.steps {
        match s {
            Step::Ffc(_) => ffc_count += 1,
            Step::ResBlock(_) => {
                res_blocks += 1;
                ffc_count += 2;
            }
            _ => {}
        }
    }
    format!("{} steps, {ffc_count} FFC_BN_ACT calls, {res_blocks} resblocks", model.steps.len())
}
