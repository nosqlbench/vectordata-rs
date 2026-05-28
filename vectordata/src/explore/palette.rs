// Copyright (c) Jonathan Shook
// SPDX-License-Identifier: Apache-2.0

//! Color palettes and intensity curves for the explore TUIs.
//!
//! Shared between the values grid (per-cell heatmap) and the unified
//! explorer's PCA scatter (per-point hue along PC4) so both surfaces
//! offer the same set of palettes — including color-blind-safe options.

/// 24-bit RGB triple. Sized to match `ratatui::style::Color::Rgb`.
pub(super) type Rgb = (u8, u8, u8);

/// Diverging-vs-sequential color palettes, including options chosen
/// for accessibility under the three common forms of color-vision
/// deficiency. Diverging palettes return `(cold, neutral, hot)`
/// anchors and bend through neutral at `t = 0.5`. Multi-stop
/// palettes (Turbo, Spectrum) override `sample()` directly with
/// their own gradient table so the eye reads many distinct hues
/// across the range — the right shape for scatter plots where
/// every bin should be unambiguous.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Palette {
    /// Classic blue ↔ red diverging. High contrast, but red↔green
    /// confusion is the most common form of color blindness — avoid
    /// for accessibility-critical displays.
    BlueRed,
    /// Blue ↔ orange diverging. ColorBrewer 'PuOr' family — safe for
    /// protan, deutan, and tritan color vision. Default for the
    /// values grid where sign matters.
    BlueOrange,
    /// Blue ↔ yellow diverging. Maximum luminance contrast at the
    /// extremes; remains distinguishable in grayscale print.
    BlueYellow,
    /// Cividis sequential. Designed for uniform perception under
    /// deuteranopia; collapses sign onto magnitude.
    Cividis,
    /// Pure monochrome ramp.
    Mono,
    /// Google's "Turbo" — a perceptually-uniform rainbow built
    /// specifically as a high-contrast replacement for the classic
    /// jet/HSV scatter palettes. Walks dark-purple → blue → teal →
    /// green → yellow → orange → dark-red, so every bin is sharply
    /// distinct. Default for the PCA scatter.
    Turbo,
    /// Saturated spectrum / hue ramp (red → orange → yellow → green →
    /// cyan → blue → violet). Maximum hue separation between bins;
    /// not perceptually uniform, but the cell colors really pop.
    Spectrum,
}

impl Palette {
    pub(super) fn label(self) -> &'static str {
        match self {
            Palette::BlueRed    => "blue-red",
            Palette::BlueOrange => "blue-orange (cb-safe)",
            Palette::BlueYellow => "blue-yellow (cb-safe)",
            Palette::Cividis    => "cividis (cb-safe)",
            Palette::Mono       => "mono",
            Palette::Turbo      => "turbo (scatter-vivid)",
            Palette::Spectrum   => "spectrum (rainbow)",
        }
    }

    pub(super) fn next(self) -> Self {
        match self {
            Palette::BlueOrange => Palette::BlueYellow,
            Palette::BlueYellow => Palette::Cividis,
            Palette::Cividis    => Palette::Mono,
            Palette::Mono       => Palette::BlueRed,
            Palette::BlueRed    => Palette::Turbo,
            Palette::Turbo      => Palette::Spectrum,
            Palette::Spectrum   => Palette::BlueOrange,
        }
    }

    /// Cold / neutral / hot anchor RGB triples for diverging palettes.
    /// Multi-stop palettes (Turbo, Spectrum) override `sample()`, so the
    /// triple returned here is only used for `is_sequential` callers
    /// that still want a coarse "extreme" color (e.g. a legend swatch).
    pub(super) fn anchors(self) -> (Rgb, Rgb, Rgb) {
        match self {
            Palette::BlueRed    => ((60, 130, 255), (160, 160, 160), (235, 80,  60)),
            Palette::BlueOrange => ((45, 110, 220), (200, 200, 200), (240, 145, 30)),
            Palette::BlueYellow => ((30,  90, 200), (170, 170, 170), (250, 220, 20)),
            Palette::Cividis    => ((  0,  35, 85),  (60,  85, 110), (255, 230, 80)),
            Palette::Mono       => ((50, 50, 50),    (130, 130, 130), (250, 250, 250)),
            // For multi-stop palettes the "anchors" are degenerate:
            // sample() handles them. The triple here is just the
            // endpoints + midpoint of the actual gradient.
            Palette::Turbo      => ((48, 18, 59),    (60, 234, 110), (122, 4, 3)),
            Palette::Spectrum   => ((220, 30, 30),   (50, 200, 80),  (140, 60, 220)),
        }
    }

    /// True for palettes that treat negative and positive identically
    /// (i.e. only magnitude matters). The values grid uses this to
    /// decide whether to flip the cold/hot anchor for sign. Turbo and
    /// Spectrum are sequential — they're scatter palettes where the
    /// whole range is one continuous hue progression.
    pub(super) fn is_sequential(self) -> bool {
        matches!(self, Palette::Cividis | Palette::Turbo | Palette::Spectrum)
    }

    /// Sample this palette at `t ∈ [0,1]`. Diverging palettes bend
    /// through neutral at `t = 0.5`; multi-stop palettes interpolate
    /// across their full gradient table.
    pub(super) fn sample(self, t: f64) -> Rgb {
        let t = t.clamp(0.0, 1.0);
        match self {
            Palette::Turbo    => sample_stops(t, &TURBO_STOPS),
            Palette::Spectrum => sample_stops(t, &SPECTRUM_STOPS),
            _ => {
                let (cold, neutral, hot) = self.anchors();
                if t < 0.5 {
                    let u = (0.5 - t) / 0.5;
                    lerp_rgb(neutral, cold, u)
                } else {
                    let u = (t - 0.5) / 0.5;
                    lerp_rgb(neutral, hot, u)
                }
            }
        }
    }
}

/// Google "Turbo" 9-stop approximation. Built from the published
/// stops (https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html);
/// piecewise-linear interpolation between these is visually close to
/// the polynomial original at every step a 16-bin scatter cares about.
const TURBO_STOPS: [(f64, Rgb); 9] = [
    (0.000, ( 48,  18,  59)),
    (0.125, ( 75,  71, 218)),
    (0.250, ( 52, 137, 238)),
    (0.375, ( 24, 200, 196)),
    (0.500, ( 60, 234, 110)),
    (0.625, (180, 245,  50)),
    (0.750, (252, 196,  50)),
    (0.875, (244, 110,  18)),
    (1.000, (122,   4,   3)),
];

/// Saturated rainbow / spectrum stops — full HSV at S=1, V=1, hue
/// rotated through 0..300° so neighbouring bins are maximally distinct.
const SPECTRUM_STOPS: [(f64, Rgb); 7] = [
    (0.000, (220,  30,  30)), // red
    (0.167, (240, 140,  20)), // orange
    (0.333, (240, 220,  20)), // yellow
    (0.500, ( 50, 200,  80)), // green
    (0.667, ( 30, 200, 220)), // cyan
    (0.833, ( 70,  90, 230)), // blue
    (1.000, (140,  60, 220)), // violet
];

fn sample_stops(t: f64, stops: &[(f64, Rgb)]) -> Rgb {
    if t <= stops[0].0 { return stops[0].1; }
    if t >= stops[stops.len() - 1].0 { return stops[stops.len() - 1].1; }
    // Find the bracketing pair via linear scan — tables are tiny.
    for w in stops.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t >= t0 && t <= t1 {
            let span = (t1 - t0).max(f64::EPSILON);
            return lerp_rgb(c0, c1, (t - t0) / span);
        }
    }
    stops[stops.len() - 1].1
}

/// Reshapes a normalized magnitude `t ∈ [0,1]` before palette lookup.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Curve {
    /// Identity.
    Linear,
    /// `sqrt(t)`. Boosts the low end.
    Sqrt,
    /// `t²`. Suppresses the low end.
    Square,
    /// Logistic centered on `t = 0.5`.
    Sigmoid,
}

impl Curve {
    pub(super) fn label(self) -> &'static str {
        match self {
            Curve::Linear  => "linear",
            Curve::Sqrt    => "sqrt",
            Curve::Square  => "square",
            Curve::Sigmoid => "sigmoid",
        }
    }
    pub(super) fn next(self) -> Self {
        match self {
            Curve::Linear  => Curve::Sqrt,
            Curve::Sqrt    => Curve::Square,
            Curve::Square  => Curve::Sigmoid,
            Curve::Sigmoid => Curve::Linear,
        }
    }
    pub(super) fn apply(self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Curve::Linear  => t,
            Curve::Sqrt    => t.sqrt(),
            Curve::Square  => t * t,
            Curve::Sigmoid => {
                let k = 12.0_f64;
                let raw = |x: f64| 1.0 / (1.0 + (-k * (x - 0.5)).exp());
                let lo = raw(0.0);
                let hi = raw(1.0);
                ((raw(t) - lo) / (hi - lo)).clamp(0.0, 1.0)
            }
        }
    }
}

fn lerp_rgb(a: Rgb, b: Rgb, t: f64) -> Rgb {
    let blend = |x: u8, y: u8| {
        let xf = x as f64;
        let yf = y as f64;
        (xf + (yf - xf) * t).round().clamp(0.0, 255.0) as u8
    };
    (blend(a.0, b.0), blend(a.1, b.1), blend(a.2, b.2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_endpoints_match_anchors() {
        for p in [Palette::BlueRed, Palette::BlueOrange, Palette::BlueYellow,
                  Palette::Cividis, Palette::Mono] {
            let (cold, neutral, hot) = p.anchors();
            assert_eq!(p.sample(0.0), cold, "{:?}.sample(0)", p);
            assert_eq!(p.sample(0.5), neutral, "{:?}.sample(0.5)", p);
            assert_eq!(p.sample(1.0), hot, "{:?}.sample(1)", p);
        }
    }

    #[test]
    fn curve_endpoints_are_exact() {
        for c in [Curve::Linear, Curve::Sqrt, Curve::Square, Curve::Sigmoid] {
            assert!((c.apply(0.0) - 0.0).abs() < 1e-9);
            assert!((c.apply(1.0) - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn palette_cycle_visits_all() {
        let mut seen = std::collections::HashSet::new();
        let mut p = Palette::BlueOrange;
        // 7 palettes — cycle for 8 steps to verify it loops.
        for _ in 0..8 { seen.insert(format!("{:?}", p)); p = p.next(); }
        assert_eq!(seen.len(), 7);
    }

    #[test]
    fn turbo_endpoints_and_midpoint() {
        let p = Palette::Turbo;
        assert_eq!(p.sample(0.0), (48, 18, 59));
        assert_eq!(p.sample(1.0), (122, 4, 3));
        // Midpoint should land on the green stop, not muddy through
        // a fake neutral.
        assert_eq!(p.sample(0.5), (60, 234, 110));
    }

    #[test]
    fn spectrum_neighbouring_bins_are_distinct() {
        // The whole point of spectrum/turbo for scatter is that
        // adjacent bin colors are unmistakably different. Quantify it
        // with a Manhattan distance threshold across the whole range.
        let n = 16;
        let p = Palette::Spectrum;
        for i in 1..n {
            let a = p.sample((i - 1) as f64 / (n - 1) as f64);
            let b = p.sample(i as f64 / (n - 1) as f64);
            let d = (a.0 as i32 - b.0 as i32).abs()
                  + (a.1 as i32 - b.1 as i32).abs()
                  + (a.2 as i32 - b.2 as i32).abs();
            assert!(d > 30, "spectrum bins {} and {} too close: {a:?} vs {b:?} (d={d})", i-1, i);
        }
    }
}
