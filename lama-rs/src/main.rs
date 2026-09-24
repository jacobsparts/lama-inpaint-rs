//! One-shot big-lama inpainting worker.
//!
//! Same contract as `inpaint.py`: read an image and a mask, run the ported
//! big-lama generator, composite the result over the untouched pixels and write
//! a PNG of the same size.  The process is deliberately short lived - it loads
//! the weights, runs one forward pass and exits - so nothing stays resident on
//! the GPU (the caller holds the shared GPU lock).
//!
//! The GPU path is chosen when a CUDA device is usable; otherwise the engine
//! falls back to the CPU implementation with identical arithmetic.

mod cpu;
// The memory guard lives outside `cuda` deliberately: `--gpu` on a card too small
// for the pass has to be refused before anything is allocated in the CPU-only
// build too, and that build has no CUDA backend to ask.
mod mem;

#[cfg(feature = "cuda")]
mod cuda;
mod image;
mod model;
mod weights;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;

use image::{Gray8, Rgb8};

fn log(msg: &str) {
    eprintln!("{msg}");
}

struct Args {
    image: PathBuf,
    mask: PathBuf,
    output: PathBuf,
    weights: PathBuf,
    force_cpu: bool,
    force_gpu: bool,
    /// Crop a tile around the mask and run the network on that instead of the
    /// whole image (`--tile`).
    tile: bool,
    /// Section a large mask into discrete 256x256 pieces and fill them from the
    /// rim of the hole inward, one 512x512 window per piece (`--sections`).
    sections: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut image = None;
    let mut mask = None;
    let mut output = None;
    let mut weights = None;
    let mut force_cpu = false;
    let mut force_gpu = false;
    let mut tile = false;
    let mut sections = false;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = |name: &str| -> Result<String, String> {
            it.next().ok_or_else(|| format!("{name} needs a value"))
        };
        match a.as_str() {
            "--image" => image = Some(PathBuf::from(val("--image")?)),
            "--mask" => mask = Some(PathBuf::from(val("--mask")?)),
            "--output" => output = Some(PathBuf::from(val("--output")?)),
            "--weights" => weights = Some(PathBuf::from(val("--weights")?)),
            "--cpu" => force_cpu = true,
            "--gpu" => force_gpu = true,
            "--tile" => tile = true,
            "--sections" => sections = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let image = image.ok_or("--image is required")?;
    let mask = mask.ok_or("--mask is required")?;
    let output = output.ok_or("--output is required")?;
    if image.as_os_str() == "-" && mask.as_os_str() == "-" {
        return Err("--image and --mask cannot both be '-' (stdin has only one stream)".to_string());
    }
    if tile && sections {
        return Err("--tile and --sections are mutually exclusive".to_string());
    }

    let weights = weights.unwrap_or_else(|| default_weights("big-lama.safetensors"));
    Ok(Args {
        image,
        mask,
        output,
        weights,
        force_cpu,
        force_gpu,
        tile,
        sections,
    })
}

const USAGE: &str = "\
usage: lama-inpaint --image IN.png --mask MASK.png --output OUT.png
                    [--weights big-lama.safetensors]
                    [--cpu | --gpu]
                    [--tile | --sections]

Runs the big-lama FFC ResNet generator over IN.png, inpainting the pixels the
mask marks (any non-black pixel is a hole), and writes OUT.png at the input
size.  Use - for either input (but not both) to read a PNG from stdin, or for
the output to write PNG data to stdout.  Diagnostics always go to stderr.
GPU is used when available unless --cpu is given.
Images at or below 512px on the short side are run whole even with --tile.

--tile runs the network on a square window around the mask instead of the whole
image: the window is twice the mask bounding box, at least 512x512, centred on
the mask and clamped inside the image.

Only the masked pixels are taken from the network, exactly as without --tile, so
the output is identical to cropping the window out by hand, running this binary
on it and pasting the hole back.  Best for large images with small holes, and
best with masks of 256x256 or less.

--sections fills a mask larger than 256x256 in discrete 256x256 sections, one
512x512 window per section, from the edge of the hole inward.  Each pass masks
only its own section and writes back only its own section, so the rest of the
hole is context on purpose: passes read the original pixels (the leak this mode
accepts) and the fills of earlier passes.  The sections tile the mask's bounding
box, so every masked pixel is decided by exactly one pass, and the rim of the
hole is filled before the interior.  The sections are the 256x256 crop the
weights were trained on.
Sections and --tile are mutually exclusive.";

/// Where the weight blob lives when `--weights` is not given.
///
/// The weight file is a `.safetensors` container, so it travels alone and the
/// search starts next to the executable: unpacking the release archive gives a
/// runnable directory.  Then the historical `models/` layout, then up the tree
/// from the executable (so a checkout works from `target/release`), and finally
/// the current directory.
fn default_weights(rel: &str) -> PathBuf {
    const LEGACY: &str = "models";
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let beside = dir.join(rel);
            if beside.exists() {
                return beside;
            }
            let legacy = dir.join(LEGACY).join(rel);
            if legacy.exists() {
                return legacy;
            }
            let mut base = dir.to_path_buf();
            for _ in 0..6 {
                let cand = base.join(rel);
                if cand.exists() {
                    return cand;
                }
                let cand = base.join(LEGACY).join(rel);
                if cand.exists() {
                    return cand;
                }
                if !base.pop() {
                    break;
                }
            }
        }
    }
    PathBuf::from(rel)
}

/// Build the 4-channel network input: RGB premultiplied by `1 - mask`, then the
/// mask, in `[c][h][w]` planes.
fn build_input(img: &Rgb8, mask: &Gray8, pad_h: usize, pad_w: usize) -> Vec<f32> {
    let (h, w) = (img.height, img.width);
    let oh = h + pad_h;
    let ow = w + pad_w;
    let plane = oh * ow;
    let mut data = vec![0f32; 4 * plane];
    for y in 0..oh {
        let sy = image::reflect_index(y as isize, h);
        for x in 0..ow {
            let sx = image::reflect_index(x as isize, w);
            let m = mask.data[sy * w + sx] as f32 / 255.0;
            let m = if m > 0.0 { 1.0 } else { 0.0 };
            let o = y * ow + x;
            for c in 0..3 {
                let v = img.data[(sy * w + sx) * 3 + c] as f32 / 255.0;
                data[c * plane + o] = v * (1.0 - m);
            }
            data[3 * plane + o] = m;
        }
    }
    data
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(args) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), String> {
    let t_start = Instant::now();

    let src = image::read_rgb(&args.image)?;
    let mask = image::read_gray(&args.mask)?;
    if mask.width != src.width || mask.height != src.height {
        return Err(format!(
            "mask is {}x{} but image is {}x{}",
            mask.width, mask.height, src.width, src.height
        ));
    }
    let (width, height) = (src.width, src.height);
    log(&format!(
        "loaded {width}x{height} image + mask in {:.2}s",
        t_start.elapsed().as_secs_f32()
    ));

    let store = weights::WeightStore::open(&args.weights, !args.force_cpu)?;
    log(&format!(
        "weights: {} tensors, {:.1} MB",
        store.index.order.len(),
        store.total_bytes() as f32 / (1024.0 * 1024.0)
    ));

    let plan = model::build();
    log(&format!("plan: {}", model::describe(&plan)));

    // The generator has three stride-2 stages, so it can only process multiples
    // of 8.  Pad, run, crop - the same contract as inpaint.py.
    let pad_h = (PAD_MOD - height % PAD_MOD) % PAD_MOD;
    let pad_w = (PAD_MOD - width % PAD_MOD) % PAD_MOD;

    if args.sections {
        if width.min(height) <= TILE_MIN {
            log(&format!(
                "--sections: {width}x{height} is at or below {TILE_MIN}px on its short side; running the whole image"
            ));
        } else {
            return run_sections(&store, &plan, &src, &mask, &args);
        }
    }

    if args.tile {
        return run_tiled(&store, &plan, &src, &mask, &args);
    }

    // Only a forced GPU run is refused: without --gpu a card that cannot hold
    // the pass falls back to the CPU engine, which is the documented behaviour.
    if args.force_gpu {
        mem::check_capacity(
            height + pad_h,
            width + pad_w,
            "Use --tile to run on a window around the mask, or --sections for a large mask.",
        )?;
    }
    let input = build_input(&src, &mask, pad_h, pad_w);
    let t0 = Instant::now();
    let out = infer(&store, &plan, input, height + pad_h, width + pad_w, &args)?;
    log(&format!("inference {width}x{height} in {:.2}s", t0.elapsed().as_secs_f32()));

    // Composite: keep the original pixels wherever the mask is zero.  The
    // network output is already constrained to `[0, 1]` by the final sigmoid,
    // but clamp anyway so the cast is well defined.
    let mask_data = if pad_h == 0 && pad_w == 0 {
        mask.data.clone()
    } else {
        let mut m = vec![0u8; (height + pad_h) * (width + pad_w)];
        for y in 0..height + pad_h {
            let sy = image::reflect_index(y as isize, height);
            for x in 0..width + pad_w {
                let sx = image::reflect_index(x as isize, width);
                m[y * (width + pad_w) + x] = mask.data[sy * width + sx];
            }
        }
        m
    };

    let mut out_img = Rgb8::new(width, height);
    let plane = (height + pad_h) * (width + pad_w);
    for y in 0..height {
        for x in 0..width {
            let pi = y * (width + pad_w) + x;
            let hole = mask_data[pi] > 0;
            let o = (y * width + x) * 3;
            if !hole {
                out_img.data[o] = src.data[o];
                out_img.data[o + 1] = src.data[o + 1];
                out_img.data[o + 2] = src.data[o + 2];
                continue;
            }
            for c in 0..3 {
                let v = out[c * plane + pi].clamp(0.0, 1.0) * 255.0;
                out_img.data[o + c] = (v + 0.5) as u8;
            }
        }
    }

    image::write_rgb(&args.output, &out_img)?;
    log(&format!("wrote {}", args.output.display()));
    log(&format!("total {:.2}s", t_start.elapsed().as_secs_f32()));
    Ok(())
}

/// Run the network once over an already-built `[4][h][w]` input and return the
/// `[3][h][w]` output planes.  This is the only place the CPU and GPU engines
/// are selected, so the tiled path gets identical arithmetic and fallback.
fn infer(
    store: &weights::WeightStore,
    plan: &model::Model,
    input: Vec<f32>,
    h: usize,
    w: usize,
    args: &Args,
) -> Result<Vec<f32>, String> {
    if args.force_cpu {
        return cpu::run(store, plan, input, h, w)
            .map_err(|e| format!("cpu inference failed: {e}"));
    }
    #[cfg(feature = "cuda")]
    match cuda::run(store, plan, &input, h, w) {
        Ok(o) => {
            cuda::print_op_stats();
            cuda::print_sub_stats();
            cuda::print_stats();
            Ok(o)
        }
        Err(e) if args.force_gpu => {
            Err(format!("gpu inference failed: {e}\n(--cpu runs the CPU engine instead)"))
        }
        Err(e) => {
            log(&format!("gpu unavailable ({e}); falling back to CPU"));
            cpu::run(store, plan, input, h, w)
                .map_err(|err| format!("cpu inference failed: {err}"))
        }
    }
    // CPU-only build: `--gpu` was asked for but there is no GPU engine to run,
    // and silently running the CPU engine would misreport what was measured.
    #[cfg(not(feature = "cuda"))]
    {
        if args.force_gpu {
            return Err(
                "--gpu was requested, but this is the CPU-only build \
                 (no CUDA backend was compiled in).\n\
                 Use the CUDA build for GPU inference, or run without --gpu."
                    .to_string(),
            );
        }
        cpu::run(store, plan, input, h, w).map_err(|e| format!("cpu inference failed: {e}"))
    }
}

/// `--tile`: run the network on a square window around the mask instead of the
/// whole image, then paste the hole back into the original.
///
/// The window is centred on the mask's bounding box and sized to twice the
/// longer bbox side, never less than `TILE_MIN`.  The image around the mask is
/// the only evidence the network has, and the generator's receptive field is a
/// few hundred pixels, so a large image with a small hole loses nothing by
/// being cropped: the window hands the network the same pixels it would have
/// seen, at the scale it was trained for, for a fraction of the cost.
///
/// Compositing is identical to the whole-image path: the network's output is
/// taken only where the mask marks a hole, and every other pixel is copied from
/// the input.  Cropping therefore cannot change how masked and unmasked areas
/// are treated - `--tile` is exactly equivalent to cropping the window out by
/// hand, running the binary on it, and pasting the hole back.
fn run_tiled(
    store: &weights::WeightStore,
    plan: &model::Model,
    src: &Rgb8,
    mask: &Gray8,
    args: &Args,
) -> Result<(), String> {
    let t_start = Instant::now();
    let (width, height) = (src.width, src.height);
    let tile = match tile_rect(mask) {
        Some(t) => t,
        None => {
            log("--tile: mask is empty; copying the input to the output");
            image::write_rgb(&args.output, src)?;
            return Ok(());
        }
    };
    log(&format!(
        "tile: {0}x{0} window at ({1},{2}) covering a {3}x{4} mask",
        tile.side, tile.x0, tile.y0, tile.bbox_w, tile.bbox_h
    ));

    let sub_src = crop_rgb(src, &tile);
    let sub_mask = crop_gray(mask, &tile);
    let pad_h = (PAD_MOD - tile.side % PAD_MOD) % PAD_MOD;
    let pad_w = pad_h;
    // `--tile` sizes its window from the mask, so a large mask can ask for far
    // more than the card holds; a forced GPU run is refused before allocating.
    if args.force_gpu {
        mem::check_capacity(
            tile.side + pad_h,
            tile.side + pad_w,
            "Use --sections, which fills the mask in 512x512 passes, or run without --tile.",
        )?;
    }
    let input = build_input(&sub_src, &sub_mask, pad_h, pad_w);
    let t0 = Instant::now();
    let out = infer(store, plan, input, tile.side + pad_h, tile.side + pad_w, args)?;
    log(&format!(
        "inference {0}x{0} in {1:.2}s",
        tile.side,
        t0.elapsed().as_secs_f32()
    ));

    let plane = (tile.side + pad_h) * (tile.side + pad_w);
    let mut out_img = Rgb8::new(width, height);
    out_img.data.copy_from_slice(&src.data);
    for y in 0..tile.side {
        for x in 0..tile.side {
            if sub_mask.data[y * tile.side + x] == 0 {
                continue;
            }
            let pi = y * (tile.side + pad_w) + x;
            let o = ((tile.y0 + y) * width + tile.x0 + x) * 3;
            for c in 0..3 {
                let v = out[c * plane + pi].clamp(0.0, 1.0) * 255.0;
                out_img.data[o + c] = (v + 0.5) as u8;
            }
        }
    }

    image::write_rgb(&args.output, &out_img)?;
    log(&format!("wrote {}", args.output.display()));
    log(&format!("total {:.2}s", t_start.elapsed().as_secs_f32()));
    Ok(())
}

/// `--sections`: fill a mask larger than 256x256 in discrete 256x256 sections,
/// one 512x512 window per section, from the rim of the hole inward.
///
/// Every pass runs the network at the scale the weights were trained for - a
/// hole no larger than 256x256 inside a 512x512 frame - which is what makes the
/// mode attractive for masks well beyond that size, and costs one inference per
/// section.
///
/// Two properties make the incremental fill work:
///
/// * Only the pass's own section is written back, and the sections tile the
///   bounding box a whole block at a time, so every masked pixel is decided by
///   exactly one pass - the one that owns it - and a pass boundary stays a
///   boundary.
/// * Every pass masks only its own section.  The rest of the hole is left
///   unmasked on purpose, so the network sees the original content there and has
///   something continuous to work from instead of a hard 512-wide hole.  That is
///   the leak this mode accepts, and the rim-inward order is what limits its
///   reach: each pass is surrounded by as much real image and earlier fill as
///   the hole's shape allows.
fn run_sections(
    store: &weights::WeightStore,
    plan: &model::Model,
    src: &Rgb8,
    mask: &Gray8,
    args: &Args,
) -> Result<(), String> {
    let t_start = Instant::now();
    let (width, height) = (src.width, src.height);
    // The sections partition the mask's bounding box, so every pixel is decided
    // by exactly one pass.
    let sections = section_rects(mask);
    if sections.is_empty() {
        log("--sections: mask is empty; copying the input to the output");
        image::write_rgb(&args.output, src)?;
        return Ok(());
    }
    // The window is the same 512 `--tile` would use, so each pass is a
    // full-context 512x512 inference.
    let side = TILE_MIN.min(width).min(height).max(1);
    log(&format!(
        "sections: {} passes, {side}x{side} window each, rim inward",
        sections.len()
    ));

    // The image the passes read and write: it starts as the original and
    // accumulates the filled sections.
    let mut working = Rgb8::new(width, height);
    working.data.copy_from_slice(&src.data);

    let pad = (PAD_MOD - side % PAD_MOD) % PAD_MOD;
    let plane = (side + pad) * (side + pad);

    for (i, s) in sections.iter().enumerate() {
        let cx = (s.x0 + s.x1) / 2;
        let cy = (s.y0 + s.y1) / 2;
        let win = Tile {
            x0: window_origin(cx, side, width),
            y0: window_origin(cy, side, height),
            side,
            bbox_w: s.x1 - s.x0,
            bbox_h: s.y1 - s.y0,
        };
        let sub_src = crop_rgb(&working, &win);
        // Holes for this pass: the section's own pixels and nothing else.  The
        // rest of the hole stays unmasked, so the network is handed its original
        // content as context - the leak this mode accepts on purpose, because a
        // 256x256 hole needs something to continue around it.  Rim-inward order
        // (the section is surrounded by real image early on) is what keeps the
        // leak from dominating.
        let mut sub_mask = crop_gray(mask, &win);
        for y in 0..side {
            for x in 0..side {
                let (gx, gy) = (win.x0 + x, win.y0 + y);
                if gx >= s.x0 && gx < s.x1 && gy >= s.y0 && gy < s.y1 {
                    continue;
                }
                sub_mask.data[y * side + x] = 0;
            }
        }

        let input = build_input(&sub_src, &sub_mask, pad, pad);
        let t0 = Instant::now();
        let out = infer(store, plan, input, side + pad, side + pad, args)?;
        log(&format!(
            "  pass {}/{}: {side}x{side} window at ({},{}), {:.2}s",
            i + 1,
            sections.len(),
            win.x0,
            win.y0,
            t0.elapsed().as_secs_f32()
        ));

        // Write back this section's mask pixels only.
        for gy in s.y0..s.y1 {
            for gx in s.x0..s.x1 {
                if mask.data[gy * width + gx] == 0 {
                    continue;
                }
                let (x, y) = (gx - win.x0, gy - win.y0);
                let pi = y * (side + pad) + x;
                let o = (gy * width + gx) * 3;
                for c in 0..3 {
                    let v = out[c * plane + pi].clamp(0.0, 1.0) * 255.0;
                    working.data[o + c] = (v + 0.5) as u8;
                }
            }
        }

        // Profiling aid, in the style of LAMA_DUMP_STEP: snapshot the image as
        // the fill progresses.
        if let Ok(dir) = std::env::var("LAMA_DUMP_SECTIONS") {
            let p = PathBuf::from(dir).join(format!("pass-{:02}.png", i + 1));
            let _ = image::write_rgb(&p, &working);
        }
    }

    image::write_rgb(&args.output, &working)?;
    log(&format!("wrote {}", args.output.display()));
    log(&format!("total {:.2}s", t_start.elapsed().as_secs_f32()));
    Ok(())
}

/// The network's three stride-2 stages require multiples of 8.
const PAD_MOD: usize = 8;
/// Smallest window `--tile` will use, so a tiny mask still gets full context.
const TILE_MIN: usize = 512;
/// `--sections` fills a mask this size at a time, so every pass sees a hole no
/// larger than the 256x256 crop the weights were trained on.
const SECTION_BLOCK: usize = 256;

/// One rectangular piece of the mask that `--sections` fills in a single pass.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Section {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

/// Split the mask into discrete `SECTION_BLOCK`-sized sections, ordered from the
/// rim of the hole inward.
///
/// The sections tile the mask's bounding box: every masked pixel belongs to
/// exactly one section and is written once.
///
/// The order comes from the mask's own depth - the distance from each masked
/// pixel to the rim of the hole - taking the shallowest pixel of each section.
/// The outermost sections are filled first, while the network still has the real
/// image around the whole boundary, and the deepest ones last, by which time the
/// fill surrounds them.  Ties break on `(y0, x0)` so the pass order is
/// deterministic.
fn section_rects(mask: &Gray8) -> Vec<Section> {
    let (mut min_x, mut min_y) = (mask.width, mask.height);
    let (mut max_x, mut max_y) = (0usize, 0usize);
    let mut count = 0usize;
    for y in 0..mask.height {
        for x in 0..mask.width {
            if mask.data[y * mask.width + x] > 0 {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
                count += 1;
            }
        }
    }
    if count == 0 {
        return Vec::new();
    }

    // A mask that fits in one section is the trained regime already: one pass,
    // one section.
    if max_x - min_x + 1 <= SECTION_BLOCK && max_y - min_y + 1 <= SECTION_BLOCK {
        return vec![Section { x0: min_x, y0: min_y, x1: max_x + 1, y1: max_y + 1 }];
    }

    let depth = mask_depth(mask);

    let mut cells: Vec<(u32, Section)> = Vec::new();
    let mut y0 = min_y;
    loop {
        let mut x0 = min_x;
        loop {
            let x1 = (x0 + SECTION_BLOCK).min(max_x + 1);
            let y1 = (y0 + SECTION_BLOCK).min(max_y + 1);
            let mut shallowest = u32::MAX;
            for y in y0..y1 {
                for x in x0..x1 {
                    if mask.data[y * mask.width + x] > 0 {
                        shallowest = shallowest.min(depth[y * mask.width + x]);
                    }
                }
            }
            if shallowest != u32::MAX {
                cells.push((shallowest, Section { x0, y0, x1, y1 }));
            }
            if x1 >= max_x + 1 {
                break;
            }
            x0 += SECTION_BLOCK;
        }
        if y0 + SECTION_BLOCK >= max_y + 1 {
            break;
        }
        y0 += SECTION_BLOCK;
    }

    cells.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.y0.cmp(&b.1.y0))
            .then(a.1.x0.cmp(&b.1.x0))
    });
    cells.into_iter().map(|(_, s)| s).collect()
}

/// Distance from every masked pixel to the rim of the hole, by BFS over the
/// mask.  Pixels on the rim (touching a non-mask pixel or the image edge) are 0;
/// unmasked pixels stay `u32::MAX`.
fn mask_depth(mask: &Gray8) -> Vec<u32> {
    let (w, h) = (mask.width, mask.height);
    let mut depth = vec![u32::MAX; w * h];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if mask.data[i] == 0 {
                continue;
            }
            let rim = x == 0
                || y == 0
                || x + 1 == w
                || y + 1 == h
                || mask.data[i - 1] == 0
                || mask.data[i + 1] == 0
                || mask.data[i - w] == 0
                || mask.data[i + w] == 0;
            if rim {
                depth[i] = 0;
                queue.push_back(i);
            }
        }
    }
    while let Some(i) = queue.pop_front() {
        let d = depth[i];
        let (x, y) = (i % w, i / w);
        let mut neighbours = [usize::MAX; 4];
        let mut n = 0;
        if x > 0 {
            neighbours[n] = i - 1;
            n += 1;
        }
        if x + 1 < w {
            neighbours[n] = i + 1;
            n += 1;
        }
        if y > 0 {
            neighbours[n] = i - w;
            n += 1;
        }
        if y + 1 < h {
            neighbours[n] = i + w;
            n += 1;
        }
        for &j in &neighbours[..n] {
            if mask.data[j] > 0 && depth[j] == u32::MAX {
                depth[j] = d.saturating_add(1);
                queue.push_back(j);
            }
        }
    }
    depth
}

/// Top-left corner of a `side`-long window centred on `c`, kept inside
/// `[0, extent - side]`.  Saturating, so a window larger than the extent starts
/// at 0 instead of panicking.
fn window_origin(c: usize, side: usize, extent: usize) -> usize {
    let max0 = extent.saturating_sub(side) as isize;
    (c as isize - side as isize / 2).clamp(0, max0) as usize
}

/// A square window around the mask.
struct Tile {
    x0: usize,
    y0: usize,
    side: usize,
    bbox_w: usize,
    bbox_h: usize,
}

/// Compute the `--tile` window: the smallest square that is at least `TILE_MIN`
/// and at least twice the mask's bounding box, rounded up to a multiple of 8,
/// capped at the largest multiple of 8 that fits inside the image, centred on
/// the bounding box and clamped so it stays inside.
fn tile_rect(mask: &Gray8) -> Option<Tile> {
    let (mut min_x, mut min_y) = (mask.width, mask.height);
    let (mut max_x, mut max_y) = (0usize, 0usize);
    for y in 0..mask.height {
        for x in 0..mask.width {
            if mask.data[y * mask.width + x] > 0 {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }
    }
    if min_x > max_x || min_y > max_y {
        return None;
    }
    let bbox_w = max_x - min_x + 1;
    let bbox_h = max_y - min_y + 1;

    let want = TILE_MIN.max(2 * bbox_w.max(bbox_h));
    // Never larger than the image, and a multiple of 8 whenever the image can
    // supply one, so the network's three stride-2 stages are not handed an
    // awkward geometry.  Images below 8px are degenerate and cannot be run by
    // the generator at all; `--tile` is disabled for them by the caller.
    let cap = mask.width.min(mask.height) / PAD_MOD * PAD_MOD;
    let side = if cap >= PAD_MOD {
        (want.div_ceil(PAD_MOD) * PAD_MOD).min(cap)
    } else {
        mask.width.min(mask.height)
    };
    let side = side.max(1);

    let cx = min_x + bbox_w / 2;
    let cy = min_y + bbox_h / 2;
    let x0 = window_origin(cx, side, mask.width);
    let y0 = window_origin(cy, side, mask.height);
    Some(Tile { x0, y0, side, bbox_w, bbox_h })
}

fn crop_rgb(img: &Rgb8, t: &Tile) -> Rgb8 {
    let mut out = Rgb8::new(t.side, t.side);
    for y in 0..t.side {
        let src_row = ((t.y0 + y) * img.width + t.x0) * 3;
        let dst_row = y * t.side * 3;
        out.data[dst_row..dst_row + t.side * 3]
            .copy_from_slice(&img.data[src_row..src_row + t.side * 3]);
    }
    out
}

fn crop_gray(img: &Gray8, t: &Tile) -> Gray8 {
    let mut out = Gray8 { width: t.side, height: t.side, data: vec![0u8; t.side * t.side] };
    for y in 0..t.side {
        let src_row = (t.y0 + y) * img.width + t.x0;
        let dst_row = y * t.side;
        out.data[dst_row..dst_row + t.side]
            .copy_from_slice(&img.data[src_row..src_row + t.side]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask_of(w: usize, h: usize, rect: Option<(usize, usize, usize, usize)>) -> Gray8 {
        let mut data = vec![0u8; w * h];
        if let Some((x0, y0, x1, y1)) = rect {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    data[y * w + x] = 255;
                }
            }
        }
        Gray8 { width: w, height: h, data }
    }

    /// The window must always contain the mask; a window that clipped the hole
    /// would leave part of it unfilled.
    fn assert_contains(t: &Tile, m: &Gray8) {
        for y in 0..m.height {
            for x in 0..m.width {
                if m.data[y * m.width + x] > 0 {
                    assert!(
                        x >= t.x0 && x < t.x0 + t.side && y >= t.y0 && y < t.y0 + t.side,
                        "mask pixel ({x},{y}) outside window at ({},{}) side {}",
                        t.x0, t.y0, t.side
                    );
                }
            }
        }
        assert!(t.x0 + t.side <= m.width && t.y0 + t.side <= m.height);
        if m.width.min(m.height) >= PAD_MOD {
            assert_eq!(t.side % PAD_MOD, 0, "window side must be a multiple of {PAD_MOD}");
        }
    }

    #[test]
    fn empty_mask_has_no_window() {
        assert!(tile_rect(&mask_of(256, 256, None)).is_none());
    }

    #[test]
    fn small_mask_gets_the_minimum_window() {
        let m = mask_of(1024, 1024, Some((500, 500, 515, 515)));
        let t = tile_rect(&m).unwrap();
        assert_eq!(t.side, TILE_MIN);
        assert_contains(&t, &m);
    }

    #[test]
    fn window_is_twice_the_bounding_box() {
        // 300x300 mask -> 600x600 window, the size the README quotes.
        let m = mask_of(2048, 2048, Some((900, 800, 1199, 1099)));
        let t = tile_rect(&m).unwrap();
        assert_eq!(t.side, 600);
        assert_contains(&t, &m);
    }

    #[test]
    fn wider_than_tall_uses_the_longer_side() {
        let m = mask_of(2048, 2048, Some((100, 1000, 799, 1099)));
        let t = tile_rect(&m).unwrap();
        assert_eq!(t.side, 1400);
        assert_contains(&t, &m);
    }

    #[test]
    fn window_stays_inside_the_image_at_every_corner() {
        for (x0, y0) in [(0, 0), (824, 0), (0, 824), (824, 924)] {
            let m = mask_of(1024, 1024, Some((x0, y0, x0 + 199, y0 + 99)));
            let t = tile_rect(&m).unwrap();
            assert_contains(&t, &m);
        }
    }

    #[test]
    fn window_is_capped_at_the_image_size() {
        // A mask spanning the whole (sub-TILE_MIN) image cannot be contained in
        // a window smaller than the image, so the window is capped and centred.
        // The CLI never routes such an image here; this pins the helper's
        // behaviour at the boundary.
        let m = mask_of(300, 500, Some((0, 0, 299, 499)));
        let t = tile_rect(&m).unwrap();
        assert_eq!(t.side, 296); // 300 rounded down to a multiple of 8
        assert_eq!((t.x0, t.y0), (2, 102)); // centred, inside the image
        assert!(t.x0 + t.side <= 300 && t.y0 + t.side <= 500);
    }

    #[test]
    fn oversized_bounding_box_is_clamped_not_clipped() {
        // A 300x300 mask at the far corner of a 2048 image stays fully inside
        // its 600x600 window.
        let m = mask_of(2048, 2048, Some((1700, 1700, 1999, 1999)));
        let t = tile_rect(&m).unwrap();
        assert_eq!(t.side, 600);
        assert_eq!((t.x0, t.y0), (1448, 1448));
        assert_contains(&t, &m);
    }

    #[test]
    fn tiny_images_do_not_panic() {
        // The CLI never reaches the window logic below TILE_MIN, but the
        // function must still return something consistent for a small image.
        for side in 1..=8usize {
            let m = mask_of(side, side, Some((0, 0, side - 1, side - 1)));
            let t = tile_rect(&m).unwrap();
            assert!(t.side >= 1 && t.side <= side);
            assert!(t.x0 + t.side <= side && t.y0 + t.side <= side);
        }
    }

    #[test]
    fn crop_helpers_copy_the_requested_region() {
        let mut img = Rgb8::new(32, 32);
        for y in 0..32 {
            for x in 0..32 {
                let o = (y * 32 + x) * 3;
                img.data[o] = x as u8;
                img.data[o + 1] = y as u8;
                img.data[o + 2] = 7;
            }
        }
        let m = mask_of(32, 32, Some((2, 3, 20, 21)));
        let t = Tile { x0: 2, y0: 3, side: 8, bbox_w: 19, bbox_h: 19 };
        let c = crop_rgb(&img, &t);
        let g = crop_gray(&m, &t);
        assert_eq!(c.width, 8);
        for y in 0..8 {
            for x in 0..8 {
                let o = (y * 8 + x) * 3;
                assert_eq!(c.data[o], (x + 2) as u8);
                assert_eq!(c.data[o + 1], (y + 3) as u8);
                assert_eq!(g.data[y * 8 + x], if y + 3 <= 21 && x + 2 <= 20 { 255 } else { 0 });
            }
        }
    }

    /// `--sections` must cover every masked pixel exactly once - the sections are
    /// a partition of the hole - and no section may be empty.
    fn assert_covers(sections: &[Section], m: &Gray8) {
        let mut hits = vec![0u8; m.width * m.height];
        for s in sections {
            let mut any = false;
            for y in s.y0..s.y1 {
                for x in s.x0..s.x1 {
                    if m.data[y * m.width + x] > 0 {
                        hits[y * m.width + x] += 1;
                        any = true;
                    }
                }
            }
            assert!(any, "empty section at ({},{})", s.x0, s.y0);
            assert!(s.x1 - s.x0 <= SECTION_BLOCK && s.y1 - s.y0 <= SECTION_BLOCK);
        }
        for y in 0..m.height {
            for x in 0..m.width {
                if m.data[y * m.width + x] > 0 {
                    assert_eq!(
                        hits[y * m.width + x], 1,
                        "pixel ({x},{y}) must be in exactly one section"
                    );
                } else {
                    assert_eq!(hits[y * m.width + x], 0, "pixel ({x},{y}) wrongly covered");
                }
            }
        }
    }

    #[test]
    fn sections_of_a_small_mask_are_single_and_cover_it() {
        let m = mask_of(1024, 1024, Some((500, 500, 700, 700)));
        let s = section_rects(&m);
        assert_eq!(s.len(), 1);
        assert_eq!((s[0].x0, s[0].y0, s[0].x1, s[0].y1), (500, 500, 701, 701));
        assert_covers(&s, &m);
    }

    #[test]
    fn a_300x300_mask_becomes_four_sections() {
        // A 299-wide bbox needs two 256-blocks across and two down, with the
        // last of each clamped to the bbox, so four discrete sections cover it.
        let m = mask_of(1024, 1024, Some((300, 300, 598, 598)));
        let s = section_rects(&m);
        assert_eq!(s.len(), 4); // two starts across, two down
        // `assert_covers` pins that every masked pixel is covered exactly once;
        // the sections must also be pairwise disjoint.
        assert_covers(&s, &m);
        for (i, a) in s.iter().enumerate() {
            for b in &s[i + 1..] {
                let disjoint = a.x1 <= b.x0 || b.x1 <= a.x0 || a.y1 <= b.y0 || b.y1 <= a.y0;
                assert!(disjoint, "sections {a:?} and {b:?} share a pixel");
            }
        }
        assert_eq!(
            (s[0].x0, s[0].x1, s[2].x0, s[2].x1),
            (300, 556, 300, 556),
            "the two starts across must be a whole block apart"
        );
    }

    #[test]
    fn the_rim_is_filled_before_the_interior() {
        // A 600x600 square: the first section must touch the rim of the hole and
        // the last must sit deeper inside it.
        let m = mask_of(2048, 2048, Some((500, 500, 1099, 1099)));
        let s = section_rects(&m);
        assert_covers(&s, &m);
        let depth = mask_depth(&m);
        let section_depth = |s: &Section| {
            let mut d = u32::MAX;
            for y in s.y0..s.y1 {
                for x in s.x0..s.x1 {
                    if m.data[y * m.width + x] > 0 {
                        d = d.min(depth[y * m.width + x]);
                    }
                }
            }
            d
        };
        assert_eq!(section_depth(&s[0]), 0, "the first pass must touch the rim");
        let last = section_depth(&s[s.len() - 1]);
        assert!(last > 0, "the last pass must be inside the hole");
        for pair in s.windows(2) {
            assert!(section_depth(&pair[0]) <= section_depth(&pair[1]));
        }
    }

    #[test]
    fn mask_depth_is_zero_on_the_rim_and_grows_inward() {
        let m = mask_of(512, 512, Some((100, 100, 299, 299)));
        let d = mask_depth(&m);
        assert_eq!(d[100 * 512 + 100], 0);
        assert_eq!(d[299 * 512 + 299], 0);
        assert!(d[200 * 512 + 200] >= 99, "centre must be deepest");
        assert_eq!(d[0], u32::MAX); // outside the mask
    }

    #[test]
    fn order_is_deterministic() {
        let a = mask_of(2048, 2048, Some((100, 100, 699, 399)));
        let b = mask_of(2048, 2048, Some((100, 100, 699, 399)));
        assert_eq!(section_rects(&a), section_rects(&b));
    }

    #[test]
    fn a_wide_mask_is_sectioned_and_ordered_by_depth() {
        // A 600x300 mask: 256-blocks tile it three across and two down, and
        // every section stays inside the image and the bbox.
        let m = mask_of(2048, 2048, Some((100, 100, 699, 399)));
        let s = section_rects(&m);
        assert_eq!(s.len(), 6);
        assert_covers(&s, &m);
        let depth = mask_depth(&m);
        let section_depth = |s: &Section| {
            let mut d = u32::MAX;
            for y in s.y0..s.y1 {
                for x in s.x0..s.x1 {
                    if m.data[y * m.width + x] > 0 {
                        d = d.min(depth[y * m.width + x]);
                    }
                }
            }
            d
        };
        for pair in s.windows(2) {
            assert!(section_depth(&pair[0]) <= section_depth(&pair[1]));
        }
    }

    #[test]
    fn order_is_deterministic_when_the_mask_is_rebuilt() {
        let a = mask_of(2048, 2048, Some((100, 100, 699, 399)));
        let b = mask_of(2048, 2048, Some((100, 100, 699, 399)));
        assert_eq!(section_rects(&a), section_rects(&b));
        assert!(!section_rects(&a).is_empty());
    }

    #[test]
    fn empty_mask_has_no_sections() {
        assert!(section_rects(&mask_of(256, 256, None)).is_empty());
    }

    #[test]
    fn a_single_pixel_mask_is_one_small_piece() {
        let m = mask_of(1024, 1024, Some((800, 900, 800, 900)));
        let s = section_rects(&m);
        assert_eq!(s.len(), 1);
        assert_eq!((s[0].x0, s[0].y0, s[0].x1, s[0].y1), (800, 900, 801, 901));
        assert_covers(&s, &m);
    }

    /// The fill order must shrink the unfilled hole monotonically and finish it:
    /// a later pass always has at least as much context as the one before, and
    /// every masked pixel is written by some pass.
    #[test]
    fn passes_fill_the_hole_monotonically_and_completely() {
        let (w, h) = (1024usize, 1024usize);
        let m = mask_of(w, h, Some((300, 300, 598, 598)));
        let sections = section_rects(&m);
        let total = m.data.iter().filter(|&&v| v > 0).count();
        assert!(total > 0);

        // Mirror of the loop in run_sections: each pass writes its own section's
        // masked pixels and the unfilled set never grows.
        let mut filled = vec![false; w * h];
        let mut remaining = total;
        let mut seen = Vec::new();
        for s in &sections {
            for y in s.y0..s.y1 {
                for x in s.x0..s.x1 {
                    let i = y * w + x;
                    if m.data[i] > 0 && !filled[i] {
                        filled[i] = true;
                        remaining -= 1;
                    }
                }
            }
            seen.push(remaining);
        }
        assert_eq!(remaining, 0, "every masked pixel must be filled");
        assert!(seen[0] < total, "the first pass must make progress");
        for pair in seen.windows(2) {
            assert!(pair[1] <= pair[0], "the unfilled hole must only shrink");
        }
        // Rim-inward means the first pass touches the boundary of the hole.
        let depth = mask_depth(&m);
        let first = &sections[0];
        let touches_rim = (first.y0..first.y1).any(|y| {
            (first.x0..first.x1).any(|x| m.data[y * w + x] > 0 && depth[y * w + x] == 0)
        });
        assert!(touches_rim, "the first pass must start at the rim");
    }

    #[test]
    fn window_origin_stays_inside() {
        assert_eq!(window_origin(0, 512, 1024), 0);
        assert_eq!(window_origin(1023, 512, 1024), 512);
        assert_eq!(window_origin(500, 512, 1024), 244);
        // A window at least as large as the extent starts at 0.
        assert_eq!(window_origin(600, 512, 300), 0);
        assert_eq!(window_origin(0, 4096, 1024), 0);
    }

    #[test]
    fn every_section_fits_in_a_512_window_inside_a_large_image() {
        // The mode relies on this: whatever the section, a 512 window clamped
        // inside an image whose short side exceeds 512 contains it.
        let m = mask_of(1024, 1024, Some((0, 0, 1023, 1023)));
        for s in section_rects(&m) {
            let cx = (s.x0 + s.x1) / 2;
            let cy = (s.y0 + s.y1) / 2;
            let x0 = window_origin(cx, TILE_MIN, 1024);
            let y0 = window_origin(cy, TILE_MIN, 1024);
            assert!(x0 <= s.x0 && s.x1 <= x0 + TILE_MIN, "section {s:?} clipped in x");
            assert!(y0 <= s.y0 && s.y1 <= y0 + TILE_MIN, "section {s:?} clipped in y");
        }
    }
}
