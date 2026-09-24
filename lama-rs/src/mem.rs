//! Device memory the engine has to fit into.
//!
//! This is the one part of the CUDA story that is needed in a build without the
//! `cuda` feature: `--gpu` on a card too small for the pass has to be refused
//! *before* anything is allocated, on the CPU-only path as much as on the GPU
//! one, because the alternative is several seconds of work and then an
//! `cuMemAlloc: out of memory`, followed by a fall back to the CPU engine that
//! for a window that size means tens of minutes.
//!
//! The query itself goes through the toolkit, which resolves the driver with
//! `dlopen`; nothing here needs a CUDA toolkit at build time.

/// Device memory the engine needs for a forward pass over `h*w` pixels, from
/// measurements on a GTX 1080 (see the README's memory note):
///
/// | input | peak used |
/// | --- | --- |
/// | 1536x1536 (2.36 Mpx) | 2933 MiB |
/// | 2048x2048 (4.19 Mpx) | 5051 MiB |
/// | 2560x2560 (6.55 Mpx) | 7619 MiB |
///
/// A least-squares fit gives `peak = 1170 B/px + 343 MiB`, which reproduces
/// those three points to within 1%.  The fixed term is the weights (195 MB)
/// plus the context; the rest is activations and im2col patch matrices, which
/// is why it dwarfs the weights and why the window side matters so much.
const BYTES_PER_PIXEL: u64 = 1170;
/// Fixed part of the estimate above, in bytes.
const FIXED_BYTES: u64 = 343 << 20;

/// Device memory the engine is expected to need for `px` input pixels.
///
/// Pure, so the arithmetic and the guard's threshold are unit-testable without
/// a GPU.
pub fn estimated_bytes(px: u64) -> u64 {
    px.saturating_mul(BYTES_PER_PIXEL).saturating_add(FIXED_BYTES)
}

/// Refuse a forward pass that the device cannot hold, before anything is
/// allocated.
///
/// The engine's peak use grows with the input area (about 1.1 GiB per
/// megapixel - see `estimated_bytes`), so a large `--tile` window or a large
/// image on a small card reaches `cuMemAlloc: out of memory` after several
/// seconds of work, and without `--gpu` it then falls back to the CPU engine,
/// which for a window that size means tens of minutes.  Failing immediately
/// with the numbers and the flag that would fit is strictly more useful.
///
/// `hint` names the flag to suggest instead (the mode that caps its work).
pub fn check_capacity(h: usize, w: usize, hint: &str) -> Result<(), String> {
    let need = estimated_bytes(h as u64 * w as u64);
    let (free, total, name) = match device_memory() {
        Some(v) => v,
        // Cannot measure: do not second-guess a run that may well work - but say
        // so, because a guard that silently never fires is worse than none.
        None => {
            eprintln!(
                "warning: cannot read device memory; running {w}x{h} without a check"
            );
            return Ok(());
        }
    };
    // 10% headroom for fragmentation and whatever else holds the card.
    if need <= free - free / 10 {
        return Ok(());
    }
    Err(format!(
        "an inference over {w}x{h} needs about {} of device memory, but only {} \
         is free of {} on \"{name}\".  {hint}",
        mib(need),
        mib(free),
        mib(total),
    ))
}

/// Bytes as a human string, in GB or MB.
fn mib(bytes: u64) -> String {
    const GIB: u64 = 1 << 30;
    if bytes >= GIB {
        format!("{:.1} GB", bytes as f64 / GIB as f64)
    } else {
        format!("{:.0} MB", bytes as f64 / (1 << 20) as f64)
    }
}

/// Free and total device memory, plus the device name, or `None` when nothing
/// can answer (no device, no driver, ...).
///
/// `lightgpu::vm::vram` runs `cuMemGetInfo` on the *primary* context, which the
/// toolkit retains but never creates, so this answers before the engine has
/// allocated anything - the whole point of `check_capacity`.  (Ask the driver's
/// own query outside a context and it fails with `CUDA_ERROR_NOT_INITIALIZED`,
/// which is why this engine once reached for the runtime API instead.)
fn device_memory() -> Option<(u64, u64, String)> {
    let (free, total) = lightgpu::vm::vram().ok()?;
    let name = lightgpu::vm::device()
        .map(|d| d.name)
        .unwrap_or_else(|_| "the GPU".to_string());
    Some((free as u64, total as u64, name))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The estimate is a straight line through the three measured points.
    #[test]
    fn estimate_matches_the_measured_points() {
        for (side, expect_mib) in [(1536usize, 2933u64), (2048, 5051), (2560, 7619)] {
            let got = estimated_bytes(side as u64 * side as u64) / (1 << 20);
            assert!(
                got.abs_diff(expect_mib) <= expect_mib / 50,
                "{side}x{side}: estimated {got} MiB, measured {expect_mib} MiB"
            );
        }
    }

    /// The window that failed must be estimated beyond an 8 GB card, and a
    /// 512x512 pass well inside one - the two facts the guard turns on.
    #[test]
    fn the_failing_window_is_out_of_range_and_a_section_is_not() {
        let gb8 = 8192u64 << 20;
        assert!(estimated_bytes(3472 * 3472) > gb8, "3472x3472 must not fit 8 GB");
        assert!(estimated_bytes(3072 * 3072) > gb8, "3072x3072 measured OOM on 8 GB");
        assert!(estimated_bytes(2560 * 2560) < gb8, "2560x2560 measured 7619 MiB");
        let pass = estimated_bytes(512 * 512);
        assert!(pass < gb8 / 8, "a 512x512 pass must fit even a 1 GB card: {pass}");
    }

    /// The estimate must grow with area and never wrap.
    #[test]
    fn estimate_is_monotone_and_saturating() {
        let mut last = 0;
        for side in [1usize, 64, 512, 1024, 2048, 4096] {
            let v = estimated_bytes(side as u64 * side as u64);
            assert!(v > last);
            last = v;
        }
        // Absurd sizes saturate rather than wrap around to a small number.
        assert!(estimated_bytes(u64::MAX / 2) >= last);
    }

    #[test]
    fn memory_is_formatted_readably() {
        assert_eq!(mib(512 << 20), "512 MB");
        assert_eq!(mib(2 << 30), "2.0 GB");
    }
}
