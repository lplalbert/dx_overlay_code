"""Efficient, differentiable screen-camera Moire and exposure synthesis.

The reference PIMoG layer constructs a radial cosine and a linear cosine
directly. This module maps a two-dimensional display lattice through a
metadata-calibrated screen/camera homography and models the low-frequency beat
that remains after camera sampling. For two periodic lattices,

``cos(phi_display) * cos(phi_sensor)``

contains a low-frequency ``cos(phi_display - phi_sensor)`` term and a
high-frequency sum term. Optical blur and pixel integration suppress most of
the latter. Evaluating the difference phase avoids a 4x--8x supersampled LCD
renderer and is therefore suitable for an inner training loop.

Four reciprocal-lattice orders are physically scored, but only one is selected
as dominant per sample; at most one weak second order is admitted. An affine
sensor/ISP carrier is fitted at one anchor, then a correlated quadratic carrier
and the non-affine projective residual are retained, so broad, fine, fan-shaped,
and saddle-shaped fringes remain one continuous phase field. Sub-percent RGB
frequency offsets accumulate relative to the same anchor and create gradual
colour separation without independent colour waves. Lens MTF and pixel-aperture
sinc attenuation can fully suppress an unsupported order. A weak
Bayer/demosaic blend introduces chromatic aliases without replacing the optical
pattern. The older independent two-wave and pairwise-product model is retained
only as an opt-in ablation. The capture path preserves absolute optical
visibility, warps the content with the same projective residual as the lattice,
filters the combined radiance with a lens PSF, and applies ambient lighting,
CFA/noise, white balance, and colour/tone processing to the complete signal
rather than to the Moire map alone.

This is a physically inspired analytic approximation, not a complete optical
renderer or ISP.  The construction follows the screen/CFA mechanisms described
in the AIM 2019 LCDMoire pipeline and Moire Attack, but is an original,
vectorized PyTorch implementation:

https://arxiv.org/abs/1911.02498
https://arxiv.org/abs/2110.10444
https://doi.org/10.1364/JOSAA.5.001828

The module is also a self-contained image-to-image command-line program.  Its
default capture calibration is embedded, so this one Python file can be copied
without the repository's ``profiles`` directory::

    python physical_moire.py input.png output.png --mode extreme --device auto

For image corruption without camera-frame rendering, use
``EfficientScreenMoireNoise.for_image_corruption(severity="medium")`` or
``--preset image_corruption --severity medium``. This preset isolates Moire
modulation from exposure loss, content warping, blur, and full-signal ISP noise.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientScreenMoireNoise(nn.Module):
    """Native-resolution analytic screen-camera Moire and exposure layer.

    The layer accepts floating-point RGB tensors in ``[0, 1]`` with shape
    ``[B, 3, H, W]`` and preserves shape and dtype.  FP16/BF16 inputs are
    calculated in FP32.  All image-dependent operations remain differentiable.

    Normal and extreme profiles differ in exposure loss, modulation strength,
    reciprocal-order visibility, and capture-geometry perturbation. Random
    geometry is shared across RGB; display-subpixel phase and CFA-like response
    differ between channels. The default metadata path selects one dominant
    reciprocal-lattice order and only occasionally admits a weak second order.
    This avoids drawing several unrelated, equally strong frequency layers on
    one screen. The older independent two-wave approximation remains available
    as an explicit ablation and is disabled by default.

    Args:
        device: Initial device for compatibility with the other noise modules.
            Temporary tensors always follow the input tensor.
        strength_scale: Multiplier applied to the sampled modulation contrast.
        linear_light: Use exact sRGB-to-linear conversion before modulation.
            The default uses a first-order gamma-equivalent modulation in sRGB,
            which is substantially faster and closely matches the exact path
            for the configured contrast range.
        phase_scale: Spatial scale used only by the optional legacy two-wave
            phase field. The metadata lattice always runs at output resolution.
        exposure_scale: Strength of display/camera exposure loss and spatial
            illumination falloff. Zero disables darkening, one uses the
            calibrated profile, and values above one strengthen it.
        high_frequency_probability: Per-sample probability of mixing the
            full-resolution metadata-driven lattice alias. Extreme mode adds
            0.2 to a configured value below one, capped at one. The default
            one gives every image an independently sampled Moire style; zero
            disables the branch.
        interference_probability: Per-sample probability of multiplicative
            interaction in the optional legacy two-wave ablation. It has no
            effect when ``legacy_low_frequency_weight`` is zero.
        interference_scale: Multiplier for legacy nonlinear cross terms.
        severe_contrast_probability: Per-sample probability of applying a
            calibrated severe-contrast multiplier. Extreme mode increases the
            probability while zero disables severe draws in both profiles.
        capture_metadata_path: JSON capture profile containing the display and
            camera resolutions plus ``projection.screen_quad_px``. ``None``
            uses the bundled profile copied from ``capture-metadata.json``.
        geometry_jitter_scale: Multiplier for small per-sample scale, roll, and
            translation perturbations around the calibrated capture geometry.
            Zero replays the exact metadata geometry while retaining phase and
            CFA-layout randomization.
        secondary_order_probability: Probability of admitting one weak second
            reciprocal-lattice order in addition to the dominant order.
            Extreme mode adds 0.05 to a non-zero probability.
        legacy_low_frequency_weight: Weight of the older independent primary /
            secondary low-frequency wave generator. The physically consistent
            default is zero, so broad fringes are produced by Nyquist folding
            of the same metadata-calibrated lattice as the fine fringes.
        cfa_mix_scale: Multiplier for the weak Bayer/demosaic contribution.
            Zero produces an optical RGB lattice without CFA reconstruction;
            one uses the calibrated profile.
        alias_period_scale: Multiplier applied to all fine/medium/coarse target
            alias periods. One uses the calibrated multi-scale profile.
        chroma_scale: Multiplier for the luminance-preserving chroma component.
            Zero makes the lattice modulation achromatic; one uses the
            calibrated colour profile.
        lattice_visibility_floor: Optional fixed lower bound for the normalized
            visibility envelope of the selected lattice order. ``None`` samples
            the calibrated normal/extreme ranges. Zero is an ablation matching
            the former locally vanishing envelope.
        content_adaptive_strength: Strength of perceptual content adaptation in
            ``[0, 1]``. Smooth regions receive moderately stronger modulation
            and strong image edges receive moderately weaker modulation. Zero
            disables the adaptation without changing the lattice phase.
        complexity_scale: Master non-negative multiplier for the correlated
            quadratic sensor carrier and RGB frequency dispersion. Zero keeps
            the affine-carrier/constant-colour-phase ablation.
        quadratic_carrier_scale: Relative multiplier for the quadratic carrier
            only. It is multiplied by ``complexity_scale`` and can be zeroed for
            a chromatic-dispersion-only ablation.
        chromatic_dispersion_scale: Relative multiplier for the RGB frequency
            offsets only. It is multiplied by ``complexity_scale`` and can be
            zeroed for a quadratic-carrier-only ablation.
        capture_lighting_scale: Master non-negative multiplier for camera white
            balance drift, low-frequency ambient veil/glare, and the output
            tone curve. Zero exactly disables these post-capture transforms.
        white_balance_scale: Relative multiplier for colour-temperature and
            green/magenta tint drift.
        ambient_glare_scale: Relative multiplier for the broad additive ambient
            reflection field that lifts black levels locally.
        tone_curve_scale: Relative multiplier for camera/ISP gamma drift.
        content_warp_scale: Multiplier for residual projective content warping
            derived from the same metadata coordinates as the lattice. Zero
            keeps the input geometry unchanged.
        native_alias_blend: Blend in ``[0, 1]`` between the broad calibrated
            alias-period sampler and the integer-fold period implied by the
            metadata geometry. The fitted ``u`` and ``v`` carriers remain one
            correlated two-dimensional sensor lattice.
        preserve_absolute_visibility: Retain the absolute lens/aperture MTF
            attenuation after order selection, with one fixed global contrast
            calibration instead of per-image peak normalization. ``False``
            reproduces the older relative-visibility stress ablation.
        optical_psf_scale: Multiplier for the Gaussian lens PSF applied to the
            complete captured radiance after Moire modulation.
        sensor_isp_scale: Master multiplier for full-signal CFA reconstruction,
            signal-dependent sensor noise, colour correction, and tone shaping.
        full_signal_cfa_scale: Relative multiplier for residual CFA/demosaic
            processing of the complete captured signal.
        sensor_noise_scale: Relative multiplier for shot/read sensor noise.
        color_correction_scale: Relative multiplier for the near-identity ISP
            colour-correction matrix and S-shaped tone response.
        display_layout: ``mixed``, ``rgb_stripe``, ``pentile``, or ``delta``.
            Mixed mode samples one layout per image and uses layout-correlated
            reciprocal-order phase and amplitude coefficients.
        sensor_cfa: ``mixed`` or one of the four Bayer orientations ``rggb``,
            ``bggr``, ``grbg``, and ``gbrg``.
        validate_input: Check finite values and the documented ``[0, 1]``
            input range. Disable this in a trusted training loop to avoid a GPU
            synchronization on every call.
        default_is_extreme: Default attack profile used when ``forward`` or
            ``configuration_for`` does not receive an explicit
            ``is_extreme`` value. The user-facing default is the stronger
            extreme profile; pass ``False`` once at construction time for a
            persistent normal-profile layer.
        spatial_coverage: ``global``, ``local``, or ``mixed``. Local modulation
            uses randomly placed smooth, irregular regions shared across RGB.
        global_coverage_probability: Whole-image probability in mixed mode.
        region_count_range: Inclusive region-count bounds, between 1 and 8.
        region_scale_range: Region radii as fractions of image width/height.
            These are axis scales, not a guaranteed covered area fraction.
        region_softness: Fraction of each region radius used for a smooth edge.
        modulation_strength_range: Optional direct linear-light amplitude
            interval; overrides profile strength before strength_scale.
    """

    NORMAL_STRENGTH_RANGE = (0.08, 0.22)
    EXTREME_STRENGTH_RANGE = (0.18, 0.38)
    # Exposure is sampled in linear light.  These ranges therefore become
    # roughly 0.72--0.86 and 0.56--0.74 after the sRGB transfer curve, making
    # the capture loss visible without crushing ordinary inputs to black.
    NORMAL_EXPOSURE_RANGE = (0.48, 0.72)
    EXTREME_EXPOSURE_RANGE = (0.28, 0.52)
    NORMAL_VIGNETTE_RANGE = (0.05, 0.15)
    EXTREME_VIGNETTE_RANGE = (0.12, 0.30)
    NORMAL_ILLUMINATION_GRADIENT_RANGE = (0.00, 0.05)
    EXTREME_ILLUMINATION_GRADIENT_RANGE = (0.02, 0.10)
    NORMAL_WHITE_BALANCE_TEMPERATURE_RANGE = (-0.035, 0.035)
    EXTREME_WHITE_BALANCE_TEMPERATURE_RANGE = (-0.070, 0.070)
    NORMAL_WHITE_BALANCE_TINT_RANGE = (-0.015, 0.015)
    EXTREME_WHITE_BALANCE_TINT_RANGE = (-0.030, 0.030)
    NORMAL_AMBIENT_GLARE_RANGE = (0.003, 0.018)
    EXTREME_AMBIENT_GLARE_RANGE = (0.008, 0.045)
    NORMAL_AMBIENT_GLARE_SIGMA_RANGE = (0.35, 0.80)
    EXTREME_AMBIENT_GLARE_SIGMA_RANGE = (0.25, 0.65)
    AMBIENT_GLARE_FLOOR_RANGE = (0.10, 0.30)
    NORMAL_TONE_GAMMA_RANGE = (0.96, 1.04)
    EXTREME_TONE_GAMMA_RANGE = (0.90, 1.10)
    NORMAL_CONTENT_WARP_PIXELS_RANGE = (0.5, 2.0)
    EXTREME_CONTENT_WARP_PIXELS_RANGE = (1.5, 5.0)
    LATTICE_SIGMA_TO_IMAGE_PSF = 4.0
    NORMAL_FULL_SIGNAL_CFA_MIX_RANGE = (0.08, 0.18)
    EXTREME_FULL_SIGNAL_CFA_MIX_RANGE = (0.14, 0.28)
    NORMAL_SENSOR_SHOT_VARIANCE_RANGE = (0.00004, 0.00016)
    EXTREME_SENSOR_SHOT_VARIANCE_RANGE = (0.00012, 0.00048)
    NORMAL_SENSOR_READ_STD_RANGE = (0.0005, 0.0018)
    EXTREME_SENSOR_READ_STD_RANGE = (0.0010, 0.0040)
    NORMAL_COLOR_MATRIX_JITTER_RANGE = (0.004, 0.016)
    EXTREME_COLOR_MATRIX_JITTER_RANGE = (0.010, 0.035)
    NORMAL_TONE_S_CURVE_RANGE = (0.00, 0.08)
    EXTREME_TONE_S_CURVE_RANGE = (0.03, 0.18)

    NORMAL_BEAT_CYCLES_RANGE = (2.0, 16.0)
    EXTREME_BEAT_CYCLES_RANGE = (1.0, 24.0)
    SENSOR_FREQUENCY_RANGE = (0.34, 0.49)
    NORMAL_HIGH_FREQUENCY_MIX_RANGE = (0.30, 0.68)
    EXTREME_HIGH_FREQUENCY_MIX_RANGE = (0.45, 0.85)
    EXTREME_HIGH_FREQUENCY_PROBABILITY_BOOST = 0.20

    NORMAL_LATTICE_SCALE_JITTER_RANGE = (-0.035, 0.035)
    EXTREME_LATTICE_SCALE_JITTER_RANGE = (-0.080, 0.080)
    NORMAL_LATTICE_ROLL_JITTER_DEGREES_RANGE = (-1.0, 1.0)
    EXTREME_LATTICE_ROLL_JITTER_DEGREES_RANGE = (-2.5, 2.5)
    NORMAL_LATTICE_TRANSLATION_JITTER_RANGE = (-0.010, 0.010)
    EXTREME_LATTICE_TRANSLATION_JITTER_RANGE = (-0.025, 0.025)
    NORMAL_LATTICE_OPTICAL_SIGMA_RANGE = (0.08, 0.16)
    EXTREME_LATTICE_OPTICAL_SIGMA_RANGE = (0.05, 0.13)
    NORMAL_LATTICE_PIXEL_FILL_FACTOR_RANGE = (0.72, 0.90)
    EXTREME_LATTICE_PIXEL_FILL_FACTOR_RANGE = (0.66, 0.88)
    NORMAL_LATTICE_TARGET_ALIAS_PERIOD_RANGE = (7.0, 72.0)
    EXTREME_LATTICE_TARGET_ALIAS_PERIOD_RANGE = (5.0, 96.0)
    NORMAL_LATTICE_TARGET_ALIAS_PERIOD_BANDS = (
        (7.0, 16.0),
        (16.0, 36.0),
        (36.0, 72.0),
    )
    EXTREME_LATTICE_TARGET_ALIAS_PERIOD_BANDS = (
        (5.0, 14.0),
        (14.0, 40.0),
        (40.0, 96.0),
    )
    NORMAL_LATTICE_SCALE_BAND_PROBABILITIES = (0.28, 0.47, 0.25)
    EXTREME_LATTICE_SCALE_BAND_PROBABILITIES = (0.34, 0.41, 0.25)
    NORMAL_LATTICE_SECONDARY_RATIO_RANGE = (0.12, 0.22)
    EXTREME_LATTICE_SECONDARY_RATIO_RANGE = (0.15, 0.25)
    NORMAL_LATTICE_CFA_MIX_RANGE = (0.16, 0.30)
    EXTREME_LATTICE_CFA_MIX_RANGE = (0.22, 0.40)
    NORMAL_LATTICE_ACHROMATIC_MIX_RANGE = (0.10, 0.30)
    EXTREME_LATTICE_ACHROMATIC_MIX_RANGE = (0.06, 0.24)
    NORMAL_LATTICE_CHROMA_BOOST_RANGE = (1.25, 1.60)
    EXTREME_LATTICE_CHROMA_BOOST_RANGE = (1.45, 1.90)
    # This floor is applied only after physical order selection. The underlying
    # lens/aperture MTF remains floor-free, so unsupported reciprocal orders do
    # not become selection candidates merely because coverage is increased.
    NORMAL_LATTICE_VISIBILITY_FLOOR_RANGE = (0.12, 0.20)
    EXTREME_LATTICE_VISIBILITY_FLOOR_RANGE = (0.22, 0.34)
    # Quadratic phase magnitude is expressed as a fraction of the requested
    # number of alias cycles over the short image edge. This keeps local chirp
    # strength comparable across resolutions and fine/medium/coarse bands.
    NORMAL_QUADRATIC_CARRIER_FRACTION_RANGE = (0.04, 0.12)
    EXTREME_QUADRATIC_CARRIER_FRACTION_RANGE = (0.08, 0.20)
    # Fractional RGB carrier offsets. A few tenths of a percent are sufficient
    # to accumulate visible colour separation over tens of beat cycles.
    NORMAL_CHROMATIC_DISPERSION_RANGE = (0.002, 0.005)
    EXTREME_CHROMATIC_DISPERSION_RANGE = (0.004, 0.008)
    CHROMATIC_DISPERSION_BALANCE_RANGE = (0.75, 1.00)
    LATTICE_ALIAS_LOG_BANDWIDTH = 0.82
    LATTICE_MODE_RANDOMNESS = 0.28
    LATTICE_SPATIAL_GATE_TEMPERATURE = 0.55
    # One fixed display/camera contrast calibration replaces the former
    # per-image peak normalization.  Multiplying the selected order's absolute
    # MTF by a global gain keeps supported aliases visible while preserving the
    # much stronger attenuation of optically unsupported orders.  The value is
    # calibrated so the default normal profile remains visible after the full
    # radiance PSF/CFA/ISP chain.
    ABSOLUTE_MTF_CONTRAST_GAIN = 6.0
    CONTENT_EDGE_SENSITIVITY = 6.0
    CONTENT_ADAPTIVE_NEUTRAL_SMOOTHNESS = 0.35
    EXTREME_SECONDARY_ORDER_PROBABILITY_BOOST = 0.05
    RECIPROCAL_LATTICE_ORDERS = (
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0, -1.0),
    )
    DISPLAY_LAYOUTS = ("rgb_stripe", "pentile", "delta")
    DISPLAY_LAYOUT_PROBABILITIES = (0.50, 0.35, 0.15)
    # Base-axis RGB subpixel phases. Diagonal-order phases are exact sums and
    # differences, so switching layouts never introduces an unrelated wave.
    DISPLAY_LAYOUT_BASE_PHASES = (
        (
            (-2.0 * math.pi / 3.0, 0.0, 2.0 * math.pi / 3.0),
            (0.0, 0.0, 0.0),
        ),
        (
            (-0.5 * math.pi, 0.0, 0.5 * math.pi),
            (0.5 * math.pi, 0.0, 0.5 * math.pi),
        ),
        (
            (-2.0 * math.pi / 3.0, 0.0, 2.0 * math.pi / 3.0),
            (math.pi / 3.0, -math.pi / 3.0, math.pi / 3.0),
        ),
    )
    DISPLAY_LAYOUT_ORDER_GAINS = (
        (1.00, 0.78, 0.72, 0.72),
        (0.82, 0.86, 1.00, 0.92),
        (0.80, 0.80, 1.00, 0.95),
    )
    DISPLAY_LAYOUT_CHROMA_GAINS = (1.00, 1.15, 1.08)
    SENSOR_CFA_LAYOUTS = ("rggb", "bggr", "grbg", "gbrg")

    DEFAULT_CAPTURE_METADATA_PATH = (
        Path(__file__).resolve().parent
        / "profiles"
        / "capture_metadata_screen_pinhole_v1.json"
    )
    # Keep the calibrated default inside the module as well as in the editable
    # JSON profile.  The repository uses the JSON when it is available; a
    # copied standalone ``physical_moire.py`` transparently falls back to this
    # equivalent minimal profile.
    EMBEDDED_CAPTURE_METADATA = {
        "model": "physical-screen-pinhole-camera-v1",
        "screen": {
            "resolution_px": [1920, 1080],
        },
        "camera": {
            "output_resolution_px": [1920, 1080],
        },
        "projection": {
            "screen_quad_px": [
                [306.0748671348506, -22.117509365896353],
                [1932.7787766985455, 101.95243102363025],
                [1691.9558112662814, 1003.2857970708653],
                [319.064132513643, 751.8750235856387],
            ],
        },
    }

    NORMAL_SEVERE_STRENGTH_MULTIPLIER_RANGE = (1.8, 2.8)
    EXTREME_SEVERE_STRENGTH_MULTIPLIER_RANGE = (1.7, 2.5)
    EXTREME_SEVERE_CONTRAST_PROBABILITY_BOOST = 0.60
    # The slow envelope should vary the contrast, not gate the pattern into a
    # local patch.  A high floor keeps the carrier visible across the frame.
    NORMAL_VISIBILITY_FLOOR_RANGE = (0.80, 0.95)
    EXTREME_VISIBILITY_FLOOR_RANGE = (0.70, 0.90)
    NORMAL_CHANNEL_RESPONSE_RANGE = (0.75, 1.30)
    EXTREME_CHANNEL_RESPONSE_RANGE = (0.55, 1.45)

    # Cross-products add sum/difference sidebands.  Their amplitude is kept
    # comparable with (but normally below) the continuous carriers, yielding
    # clear crossings without collapsing the pattern into isolated cells.
    NORMAL_LOW_WAVE_INTERACTION_RANGE = (0.50, 0.90)
    EXTREME_LOW_WAVE_INTERACTION_RANGE = (0.75, 1.25)
    NORMAL_HIGH_WAVE_INTERACTION_RANGE = (0.40, 0.75)
    EXTREME_HIGH_WAVE_INTERACTION_RANGE = (0.60, 1.05)
    EXTREME_INTERFERENCE_PROBABILITY_BOOST = 0.15

    NORMAL_RADIAL_PIXELS_RANGE = (0.4, 2.8)
    EXTREME_RADIAL_PIXELS_RANGE = (1.5, 6.5)
    NORMAL_BEND_PIXELS_RANGE = (0.3, 2.4)
    EXTREME_BEND_PIXELS_RANGE = (1.0, 5.0)
    NORMAL_RIPPLE_PIXELS_RANGE = (0.2, 1.4)
    EXTREME_RIPPLE_PIXELS_RANGE = (0.8, 3.0)
    NORMAL_PERSPECTIVE_RANGE = (-0.018, 0.018)
    EXTREME_PERSPECTIVE_RANGE = (-0.035, 0.035)

    # Both nearly orthogonal wave families must remain visible over the whole
    # screen; a very weak secondary family looks like a single straight grid.
    SECONDARY_WEIGHT_RANGE = (0.35, 0.75)
    ACHROMATIC_MIX_RANGE = (0.22, 0.58)
    COLOR_PHASE_SCALE_RANGE = (0.50, 1.05)

    def __init__(
        self,
        device: object = "cpu",
        strength_scale: float = 1.0,
        linear_light: bool = False,
        phase_scale: float = 0.5,
        exposure_scale: float = 1.0,
        high_frequency_probability: float = 1.0,
        interference_probability: float = 0.85,
        interference_scale: float = 1.0,
        severe_contrast_probability: float = 0.0,
        capture_metadata_path: Optional[object] = None,
        geometry_jitter_scale: float = 1.0,
        secondary_order_probability: float = 0.15,
        legacy_low_frequency_weight: float = 0.0,
        cfa_mix_scale: float = 1.0,
        alias_period_scale: float = 1.0,
        chroma_scale: float = 1.0,
        lattice_visibility_floor: Optional[float] = None,
        content_adaptive_strength: float = 0.35,
        complexity_scale: float = 1.0,
        quadratic_carrier_scale: float = 1.0,
        chromatic_dispersion_scale: float = 1.0,
        capture_lighting_scale: float = 1.0,
        white_balance_scale: float = 1.0,
        ambient_glare_scale: float = 1.0,
        tone_curve_scale: float = 1.0,
        content_warp_scale: float = 1.0,
        native_alias_blend: float = 0.35,
        preserve_absolute_visibility: bool = True,
        optical_psf_scale: float = 1.0,
        sensor_isp_scale: float = 1.0,
        full_signal_cfa_scale: float = 1.0,
        sensor_noise_scale: float = 1.0,
        color_correction_scale: float = 1.0,
        display_layout: str = "mixed",
        sensor_cfa: str = "mixed",
        validate_input: bool = True,
        default_is_extreme: bool = True,
        spatial_coverage: str = "global",
        global_coverage_probability: float = 0.2,
        region_count_range: Tuple[int, int] = (1, 3),
        region_scale_range: Tuple[float, float] = (0.18, 0.48),
        region_softness: float = 0.4,
        modulation_strength_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__()
        if not isinstance(default_is_extreme, bool):
            raise TypeError("default_is_extreme must be bool")
        if spatial_coverage not in ("global", "local", "mixed"):
            raise ValueError("spatial_coverage must be 'global', 'local', or 'mixed'")
        if not math.isfinite(global_coverage_probability) or not 0 <= global_coverage_probability <= 1:
            raise ValueError("global_coverage_probability must be in [0,1]")
        if not math.isfinite(region_softness) or not 0 < region_softness <= 1:
            raise ValueError("region_softness must be in (0,1]")
        self.region_count_range = tuple(int(v) for v in self._sampling_interval(
            region_count_range, "region_count_range", 1, 8, integer=True
        ))
        self.region_scale_range = self._sampling_interval(
            region_scale_range, "region_scale_range", 0.01, 1.0
        )
        self.modulation_strength_range = (
            None if modulation_strength_range is None else self._sampling_interval(
                modulation_strength_range, "modulation_strength_range", 0.0
            )
        )
        self.spatial_coverage = spatial_coverage
        self.global_coverage_probability = float(global_coverage_probability)
        self.region_softness = float(region_softness)
        if not math.isfinite(strength_scale) or strength_scale < 0.0:
            raise ValueError("strength_scale must be finite and non-negative")
        if not math.isfinite(phase_scale) or not 0.0 < phase_scale <= 1.0:
            raise ValueError("phase_scale must be finite and in (0, 1]")
        if not math.isfinite(exposure_scale) or exposure_scale < 0.0:
            raise ValueError("exposure_scale must be finite and non-negative")
        if (
            not math.isfinite(high_frequency_probability)
            or not 0.0 <= high_frequency_probability <= 1.0
        ):
            raise ValueError("high_frequency_probability must be in [0, 1]")
        if (
            not math.isfinite(interference_probability)
            or not 0.0 <= interference_probability <= 1.0
        ):
            raise ValueError("interference_probability must be in [0, 1]")
        if not math.isfinite(interference_scale) or interference_scale < 0.0:
            raise ValueError(
                "interference_scale must be finite and non-negative"
            )
        if (
            not math.isfinite(severe_contrast_probability)
            or not 0.0 <= severe_contrast_probability <= 1.0
        ):
            raise ValueError(
                "severe_contrast_probability must be in [0, 1]"
            )
        if (
            not math.isfinite(geometry_jitter_scale)
            or geometry_jitter_scale < 0.0
        ):
            raise ValueError(
                "geometry_jitter_scale must be finite and non-negative"
            )
        if (
            not math.isfinite(secondary_order_probability)
            or not 0.0 <= secondary_order_probability <= 1.0
        ):
            raise ValueError("secondary_order_probability must be in [0, 1]")
        if (
            not math.isfinite(legacy_low_frequency_weight)
            or not 0.0 <= legacy_low_frequency_weight <= 1.0
        ):
            raise ValueError("legacy_low_frequency_weight must be in [0, 1]")
        if not math.isfinite(cfa_mix_scale) or cfa_mix_scale < 0.0:
            raise ValueError("cfa_mix_scale must be finite and non-negative")
        if not math.isfinite(alias_period_scale) or alias_period_scale <= 0.0:
            raise ValueError("alias_period_scale must be finite and positive")
        if not math.isfinite(chroma_scale) or chroma_scale < 0.0:
            raise ValueError("chroma_scale must be finite and non-negative")
        if lattice_visibility_floor is not None and (
            not math.isfinite(lattice_visibility_floor)
            or not 0.0 <= lattice_visibility_floor <= 1.0
        ):
            raise ValueError("lattice_visibility_floor must be in [0, 1]")
        if (
            not math.isfinite(content_adaptive_strength)
            or not 0.0 <= content_adaptive_strength <= 1.0
        ):
            raise ValueError("content_adaptive_strength must be in [0, 1]")
        for parameter_name, parameter_value in (
            ("complexity_scale", complexity_scale),
            ("quadratic_carrier_scale", quadratic_carrier_scale),
            ("chromatic_dispersion_scale", chromatic_dispersion_scale),
            ("capture_lighting_scale", capture_lighting_scale),
            ("white_balance_scale", white_balance_scale),
            ("ambient_glare_scale", ambient_glare_scale),
            ("tone_curve_scale", tone_curve_scale),
            ("content_warp_scale", content_warp_scale),
            ("optical_psf_scale", optical_psf_scale),
            ("sensor_isp_scale", sensor_isp_scale),
            ("full_signal_cfa_scale", full_signal_cfa_scale),
            ("sensor_noise_scale", sensor_noise_scale),
            ("color_correction_scale", color_correction_scale),
        ):
            if not math.isfinite(parameter_value) or parameter_value < 0.0:
                raise ValueError(
                    f"{parameter_name} must be finite and non-negative"
                )
        if (
            not math.isfinite(native_alias_blend)
            or not 0.0 <= native_alias_blend <= 1.0
        ):
            raise ValueError("native_alias_blend must be in [0, 1]")
        if not isinstance(preserve_absolute_visibility, bool):
            raise TypeError("preserve_absolute_visibility must be bool")
        display_layout = str(display_layout).lower()
        if display_layout != "mixed" and display_layout not in self.DISPLAY_LAYOUTS:
            raise ValueError(
                "display_layout must be 'mixed', 'rgb_stripe', 'pentile', "
                "or 'delta'"
            )
        sensor_cfa = str(sensor_cfa).lower()
        if sensor_cfa != "mixed" and sensor_cfa not in self.SENSOR_CFA_LAYOUTS:
            raise ValueError(
                "sensor_cfa must be 'mixed', 'rggb', 'bggr', 'grbg', or "
                "'gbrg'"
            )
        self.default_is_extreme = default_is_extreme
        self.strength_scale = float(strength_scale)
        self.linear_light = bool(linear_light)
        self.phase_scale = float(phase_scale)
        self.exposure_scale = float(exposure_scale)
        self.high_frequency_probability = float(high_frequency_probability)
        self.interference_probability = float(interference_probability)
        self.interference_scale = float(interference_scale)
        self.severe_contrast_probability = float(
            severe_contrast_probability
        )
        self.geometry_jitter_scale = float(geometry_jitter_scale)
        self.secondary_order_probability = float(secondary_order_probability)
        self.legacy_low_frequency_weight = float(legacy_low_frequency_weight)
        self.cfa_mix_scale = float(cfa_mix_scale)
        self.alias_period_scale = float(alias_period_scale)
        self.chroma_scale = float(chroma_scale)
        self.lattice_visibility_floor = (
            None
            if lattice_visibility_floor is None
            else float(lattice_visibility_floor)
        )
        self.content_adaptive_strength = float(content_adaptive_strength)
        self.complexity_scale = float(complexity_scale)
        self.quadratic_carrier_scale = float(quadratic_carrier_scale)
        self.chromatic_dispersion_scale = float(chromatic_dispersion_scale)
        self.capture_lighting_scale = float(capture_lighting_scale)
        self.white_balance_scale = float(white_balance_scale)
        self.ambient_glare_scale = float(ambient_glare_scale)
        self.tone_curve_scale = float(tone_curve_scale)
        self.content_warp_scale = float(content_warp_scale)
        self.native_alias_blend = float(native_alias_blend)
        self.preserve_absolute_visibility = preserve_absolute_visibility
        self.optical_psf_scale = float(optical_psf_scale)
        self.sensor_isp_scale = float(sensor_isp_scale)
        self.full_signal_cfa_scale = float(full_signal_cfa_scale)
        self.sensor_noise_scale = float(sensor_noise_scale)
        self.color_correction_scale = float(color_correction_scale)
        self.display_layout = display_layout
        self.sensor_cfa = sensor_cfa
        self.validate_input = bool(validate_input)
        if capture_metadata_path is None:
            metadata_path = self.DEFAULT_CAPTURE_METADATA_PATH
            if metadata_path.is_file():
                metadata, resolved_metadata_path = self._load_capture_metadata(
                    metadata_path
                )
                metadata_source = str(resolved_metadata_path)
            else:
                metadata = self.EMBEDDED_CAPTURE_METADATA
                metadata_source = "<embedded:physical-screen-pinhole-camera-v1>"
        else:
            metadata_path = Path(capture_metadata_path).expanduser()
            metadata, resolved_metadata_path = self._load_capture_metadata(
                metadata_path
            )
            metadata_source = str(resolved_metadata_path)
        geometry = self._capture_geometry_from_metadata(metadata)
        self.capture_metadata_path = metadata_source
        self.capture_model = str(metadata.get("model", "unknown"))
        target_device = torch.device(device)
        self.register_buffer(
            "_device_anchor",
            torch.empty(0, device=target_device),
            persistent=False,
        )
        self.register_buffer(
            "_camera_to_screen_homography",
            geometry["camera_to_screen"].to(target_device),
            persistent=False,
        )
        self.register_buffer(
            "_screen_resolution",
            geometry["screen_resolution"].to(target_device),
            persistent=False,
        )
        self.register_buffer(
            "_camera_resolution",
            geometry["camera_resolution"].to(target_device),
            persistent=False,
        )
        self.register_buffer(
            "_normalized_screen_quad",
            geometry["normalized_screen_quad"].to(target_device),
            persistent=False,
        )

    @staticmethod
    def _sampling_interval(
        value: object, name: str, lower: float,
        upper: float = math.inf, integer: bool = False,
    ) -> Tuple[float, float]:
        try:
            if isinstance(value, (str, bytes)) or len(value) != 2:
                raise ValueError
            pair = tuple(float(v) for v in value)
            if (not all(math.isfinite(v) for v in pair)
                    or not lower <= pair[0] <= pair[1] <= upper
                    or (integer and any(int(v) != v for v in pair))):
                raise ValueError
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"{name} must contain two ordered finite values in [{lower},{upper}]") from error
        return pair

    @classmethod
    def for_image_corruption(
        cls,
        severity: str = "random",
        **overrides: Any,
    ) -> "EfficientScreenMoireNoise":
        """Create a shape-preserving Moire corruption for robustness training.

        Random severity samples amplitude from [0.12,1.2] per image. The
        default mixed coverage uses local regions 80% of the time and global
        coverage 20% of the time. Light/medium/strong multiply the same sampled
        modulation by 1/2/4.
        With identical inputs, settings and generator seeds, severity changes
        amplitude only. Patterns vary across images/calls and span the existing
        fine/medium/coarse bands. Contrast is deliberately normalized for
        visible corruption rather than constrained by a calibrated camera MTF.

        ``alias_period_scale`` adjusts the target band and ``chroma_scale``
        adjusts colour fringing. Overrides use the normal constructor options;
        re-enabling exposure/warp/ISP explicitly creates a combined corruption.
        """
        strengths = {"random": 1.0, "light": 1.0, "medium": 2.0, "strong": 4.0}
        if severity not in strengths:
            raise ValueError("severity must be 'random', 'light', 'medium', or 'strong'")
        options = {
            "strength_scale": strengths[severity],
            "linear_light": True,
            "exposure_scale": 0.0,
            "capture_lighting_scale": 0.0,
            "content_warp_scale": 0.0,
            "optical_psf_scale": 0.0,
            "sensor_isp_scale": 0.0,
            "native_alias_blend": 0.0,
            "preserve_absolute_visibility": False,
            "high_frequency_probability": 1.0,
            "lattice_visibility_floor": 0.35,
            "content_adaptive_strength": 0.2,
            "secondary_order_probability": 0.3,
            "default_is_extreme": True,
            "spatial_coverage": "mixed",
            "modulation_strength_range": (0.12, 1.2) if severity == "random" else None,
        }
        options.update(overrides)
        return cls(**options)

    @property
    def device(self) -> torch.device:
        """Compatibility property reflecting the module's current device."""

        return self._device_anchor.device

    @property
    def default_profile(self) -> str:
        """Name of the profile used when a call does not override it."""

        return "extreme" if self.default_is_extreme else "normal"

    def _resolve_is_extreme(self, is_extreme: Optional[bool]) -> bool:
        if is_extreme is None:
            return self.default_is_extreme
        if not isinstance(is_extreme, bool):
            raise TypeError("is_extreme must be bool or None")
        return is_extreme

    def extra_repr(self) -> str:
        return f"default_profile={self.default_profile}, device={self.device}"

    @staticmethod
    def _load_capture_metadata(
        metadata_path: Path,
    ) -> Tuple[Mapping[str, Any], Path]:
        try:
            resolved = metadata_path.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"capture metadata does not exist: {metadata_path}"
            ) from error
        try:
            with resolved.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"capture metadata is not valid JSON: {resolved}"
            ) from error
        if not isinstance(metadata, Mapping):
            raise ValueError("capture metadata root must be a JSON object")
        return metadata, resolved

    @staticmethod
    def _positive_resolution(
        value: object,
        field_name: str,
    ) -> Tuple[float, float]:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 2
        ):
            raise ValueError(f"{field_name} must contain width and height")
        width, height = float(value[0]), float(value[1])
        if (
            not math.isfinite(width)
            or not math.isfinite(height)
            or width <= 1.0
            or height <= 1.0
        ):
            raise ValueError(f"{field_name} values must be finite and > 1")
        return width, height

    @staticmethod
    def _unit_square_to_quad_homography(
        normalized_quad: torch.Tensor,
    ) -> torch.Tensor:
        source = normalized_quad.new_tensor(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        )
        rows = []
        values = []
        for index in range(4):
            u, v = source[index]
            x, y = normalized_quad[index]
            zero = u.new_zeros(())
            one = u.new_ones(())
            rows.append(
                torch.stack((u, v, one, zero, zero, zero, -x * u, -x * v))
            )
            rows.append(
                torch.stack((zero, zero, zero, u, v, one, -y * u, -y * v))
            )
            values.extend((x, y))
        matrix = torch.stack(rows)
        target = torch.stack(values)
        try:
            coefficients = torch.linalg.solve(matrix, target)
        except RuntimeError as error:
            raise ValueError("projection.screen_quad_px is degenerate") from error
        homography = torch.cat(
            (coefficients, coefficients.new_ones(1)),
        ).view(3, 3)
        if (
            not bool(torch.isfinite(homography).all())
            or float(torch.linalg.det(homography).abs()) < 1e-10
        ):
            raise ValueError("projection.screen_quad_px is degenerate")
        return homography

    @classmethod
    def _capture_geometry_from_metadata(
        cls,
        metadata: Mapping[str, Any],
    ) -> Dict[str, torch.Tensor]:
        try:
            screen = metadata["screen"]
            camera = metadata["camera"]
            projection = metadata["projection"]
            screen_resolution_value = screen["resolution_px"]
            camera_resolution_value = camera["output_resolution_px"]
            quad_value = projection["screen_quad_px"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "capture metadata must define screen.resolution_px, "
                "camera.output_resolution_px, and "
                "projection.screen_quad_px"
            ) from error

        screen_width, screen_height = cls._positive_resolution(
            screen_resolution_value,
            "screen.resolution_px",
        )
        camera_width, camera_height = cls._positive_resolution(
            camera_resolution_value,
            "camera.output_resolution_px",
        )
        try:
            quad = torch.tensor(quad_value, dtype=torch.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "projection.screen_quad_px must be a numeric 4x2 array"
            ) from error
        if quad.shape != (4, 2) or not bool(torch.isfinite(quad).all()):
            raise ValueError(
                "projection.screen_quad_px must be a finite 4x2 array"
            )
        normalized_quad = quad.clone()
        normalized_quad[:, 0] /= camera_width - 1.0
        normalized_quad[:, 1] /= camera_height - 1.0
        screen_to_camera = cls._unit_square_to_quad_homography(
            normalized_quad
        )
        return {
            "camera_to_screen": torch.linalg.inv(screen_to_camera).float(),
            "screen_resolution": torch.tensor(
                (screen_width, screen_height), dtype=torch.float32
            ),
            "camera_resolution": torch.tensor(
                (camera_width, camera_height), dtype=torch.float32
            ),
            "normalized_screen_quad": normalized_quad.float(),
        }

    def _validate_image(self, img: torch.Tensor) -> None:
        if img.ndim != 4:
            raise ValueError(
                "EfficientScreenMoireNoise expects [B, 3, H, W], got "
                f"{tuple(img.shape)}"
            )
        batch, channels, height, width = img.shape
        if batch <= 0:
            raise ValueError("input batch must be non-empty")
        if channels != 3:
            raise ValueError(
                f"EfficientScreenMoireNoise supports RGB only, got C={channels}"
            )
        if height < 2 or width < 2:
            raise ValueError("input height and width must both be at least 2")
        if not img.is_floating_point():
            raise TypeError("input image must be a floating-point tensor")
        if self.validate_input:
            detached = img.detach()
            tolerance = 1e-4
            invalid = (
                ~torch.isfinite(detached)
                | (detached < -tolerance)
                | (detached > 1.0 + tolerance)
            )
            if bool(invalid.any()):
                if not bool(torch.isfinite(detached).all()):
                    raise ValueError("input image contains NaN or Inf")
                minimum = float(detached.amin().item())
                maximum = float(detached.amax().item())
                raise ValueError(
                    "input image must be in [0, 1], got range "
                    f"[{minimum:.6g}, {maximum:.6g}]"
                )

    @staticmethod
    def _working_image(img: torch.Tensor) -> torch.Tensor:
        if img.dtype in (torch.float16, torch.bfloat16):
            return img.float()
        return img

    @staticmethod
    def _uniform(
        reference: torch.Tensor,
        low: float,
        high: float,
        shape: Tuple[int, ...],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        random = torch.rand(
            shape,
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )
        return low + (high - low) * random

    @classmethod
    def _signed_uniform(
        cls,
        reference: torch.Tensor,
        low: float,
        high: float,
        shape: Tuple[int, ...],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        magnitude = cls._uniform(reference, low, high, shape, generator)
        selector = cls._uniform(reference, 0.0, 1.0, shape, generator)
        sign = selector.ge(0.5).to(dtype=reference.dtype).mul(2.0).sub(1.0)
        return magnitude * sign

    @staticmethod
    def _srgb_to_linear(img: torch.Tensor) -> torch.Tensor:
        return torch.where(
            img <= 0.04045,
            img / 12.92,
            ((img + 0.055) / 1.055).pow(2.4),
        )

    @staticmethod
    def _linear_to_srgb(img: torch.Tensor) -> torch.Tensor:
        # Clamp the unused power branch away from zero. ``torch.where``
        # evaluates both branches, and x**(1/2.4) has an infinite derivative at
        # zero that would otherwise contaminate FP16 gradients with NaNs.
        power_input = img.clamp_min(0.0031308)
        return torch.where(
            img <= 0.0031308,
            img * 12.92,
            1.055 * power_input.pow(1.0 / 2.4) - 0.055,
        )

    @staticmethod
    def _pixel_grid(
        reference: torch.Tensor,
        height: int,
        width: int,
        sample_height: int,
        sample_width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y = torch.linspace(
            0.0,
            float(height - 1),
            steps=sample_height,
            device=reference.device,
            dtype=reference.dtype,
        ).view(1, 1, sample_height, 1)
        x = torch.linspace(
            0.0,
            float(width - 1),
            steps=sample_width,
            device=reference.device,
            dtype=reference.dtype,
        ).view(1, 1, 1, sample_width)
        centered_x = x - (float(width - 1) / 2.0)
        centered_y = y - (float(height - 1) / 2.0)
        normalized_x = centered_x / max(float(width - 1) / 2.0, 0.5)
        normalized_y = centered_y / max(float(height - 1) / 2.0, 0.5)
        return centered_x, centered_y, normalized_x, normalized_y

    @classmethod
    def _profile_ranges(
        cls,
        is_extreme: bool,
    ) -> Dict[str, Tuple[float, float]]:
        if is_extreme:
            return {
                "strength": cls.EXTREME_STRENGTH_RANGE,
                "exposure": cls.EXTREME_EXPOSURE_RANGE,
                "vignette": cls.EXTREME_VIGNETTE_RANGE,
                "illumination_gradient": (
                    cls.EXTREME_ILLUMINATION_GRADIENT_RANGE
                ),
                "white_balance_temperature": (
                    cls.EXTREME_WHITE_BALANCE_TEMPERATURE_RANGE
                ),
                "white_balance_tint": (
                    cls.EXTREME_WHITE_BALANCE_TINT_RANGE
                ),
                "ambient_glare": cls.EXTREME_AMBIENT_GLARE_RANGE,
                "ambient_glare_sigma": (
                    cls.EXTREME_AMBIENT_GLARE_SIGMA_RANGE
                ),
                "tone_gamma": cls.EXTREME_TONE_GAMMA_RANGE,
                "content_warp_pixels": (
                    cls.EXTREME_CONTENT_WARP_PIXELS_RANGE
                ),
                "full_signal_cfa_mix": (
                    cls.EXTREME_FULL_SIGNAL_CFA_MIX_RANGE
                ),
                "sensor_shot_variance": (
                    cls.EXTREME_SENSOR_SHOT_VARIANCE_RANGE
                ),
                "sensor_read_std": cls.EXTREME_SENSOR_READ_STD_RANGE,
                "color_matrix_jitter": (
                    cls.EXTREME_COLOR_MATRIX_JITTER_RANGE
                ),
                "tone_s_curve": cls.EXTREME_TONE_S_CURVE_RANGE,
                "beat_cycles": cls.EXTREME_BEAT_CYCLES_RANGE,
                "high_mix": cls.EXTREME_HIGH_FREQUENCY_MIX_RANGE,
                "lattice_scale_jitter": (
                    cls.EXTREME_LATTICE_SCALE_JITTER_RANGE
                ),
                "lattice_roll_jitter_degrees": (
                    cls.EXTREME_LATTICE_ROLL_JITTER_DEGREES_RANGE
                ),
                "lattice_translation_jitter": (
                    cls.EXTREME_LATTICE_TRANSLATION_JITTER_RANGE
                ),
                "lattice_optical_sigma": (
                    cls.EXTREME_LATTICE_OPTICAL_SIGMA_RANGE
                ),
                "lattice_pixel_fill_factor": (
                    cls.EXTREME_LATTICE_PIXEL_FILL_FACTOR_RANGE
                ),
                "lattice_target_alias_period": (
                    cls.EXTREME_LATTICE_TARGET_ALIAS_PERIOD_RANGE
                ),
                "lattice_secondary_ratio": (
                    cls.EXTREME_LATTICE_SECONDARY_RATIO_RANGE
                ),
                "lattice_cfa_mix": (
                    cls.EXTREME_LATTICE_CFA_MIX_RANGE
                ),
                "lattice_achromatic_mix": (
                    cls.EXTREME_LATTICE_ACHROMATIC_MIX_RANGE
                ),
                "lattice_chroma_boost": (
                    cls.EXTREME_LATTICE_CHROMA_BOOST_RANGE
                ),
                "lattice_visibility_floor": (
                    cls.EXTREME_LATTICE_VISIBILITY_FLOOR_RANGE
                ),
                "quadratic_carrier_fraction": (
                    cls.EXTREME_QUADRATIC_CARRIER_FRACTION_RANGE
                ),
                "chromatic_dispersion": (
                    cls.EXTREME_CHROMATIC_DISPERSION_RANGE
                ),
                "severe_strength_multiplier": (
                    cls.EXTREME_SEVERE_STRENGTH_MULTIPLIER_RANGE
                ),
                "visibility_floor": cls.EXTREME_VISIBILITY_FLOOR_RANGE,
                "channel_response": cls.EXTREME_CHANNEL_RESPONSE_RANGE,
                "low_wave_interaction": (
                    cls.EXTREME_LOW_WAVE_INTERACTION_RANGE
                ),
                "high_wave_interaction": (
                    cls.EXTREME_HIGH_WAVE_INTERACTION_RANGE
                ),
                "radial_pixels": cls.EXTREME_RADIAL_PIXELS_RANGE,
                "bend_pixels": cls.EXTREME_BEND_PIXELS_RANGE,
                "ripple_pixels": cls.EXTREME_RIPPLE_PIXELS_RANGE,
                "perspective": cls.EXTREME_PERSPECTIVE_RANGE,
            }
        return {
            "strength": cls.NORMAL_STRENGTH_RANGE,
            "exposure": cls.NORMAL_EXPOSURE_RANGE,
            "vignette": cls.NORMAL_VIGNETTE_RANGE,
            "illumination_gradient": cls.NORMAL_ILLUMINATION_GRADIENT_RANGE,
            "white_balance_temperature": (
                cls.NORMAL_WHITE_BALANCE_TEMPERATURE_RANGE
            ),
            "white_balance_tint": cls.NORMAL_WHITE_BALANCE_TINT_RANGE,
            "ambient_glare": cls.NORMAL_AMBIENT_GLARE_RANGE,
            "ambient_glare_sigma": cls.NORMAL_AMBIENT_GLARE_SIGMA_RANGE,
            "tone_gamma": cls.NORMAL_TONE_GAMMA_RANGE,
            "content_warp_pixels": cls.NORMAL_CONTENT_WARP_PIXELS_RANGE,
            "full_signal_cfa_mix": cls.NORMAL_FULL_SIGNAL_CFA_MIX_RANGE,
            "sensor_shot_variance": (
                cls.NORMAL_SENSOR_SHOT_VARIANCE_RANGE
            ),
            "sensor_read_std": cls.NORMAL_SENSOR_READ_STD_RANGE,
            "color_matrix_jitter": cls.NORMAL_COLOR_MATRIX_JITTER_RANGE,
            "tone_s_curve": cls.NORMAL_TONE_S_CURVE_RANGE,
            "beat_cycles": cls.NORMAL_BEAT_CYCLES_RANGE,
            "high_mix": cls.NORMAL_HIGH_FREQUENCY_MIX_RANGE,
            "lattice_scale_jitter": cls.NORMAL_LATTICE_SCALE_JITTER_RANGE,
            "lattice_roll_jitter_degrees": (
                cls.NORMAL_LATTICE_ROLL_JITTER_DEGREES_RANGE
            ),
            "lattice_translation_jitter": (
                cls.NORMAL_LATTICE_TRANSLATION_JITTER_RANGE
            ),
            "lattice_optical_sigma": cls.NORMAL_LATTICE_OPTICAL_SIGMA_RANGE,
            "lattice_pixel_fill_factor": (
                cls.NORMAL_LATTICE_PIXEL_FILL_FACTOR_RANGE
            ),
            "lattice_target_alias_period": (
                cls.NORMAL_LATTICE_TARGET_ALIAS_PERIOD_RANGE
            ),
            "lattice_secondary_ratio": (
                cls.NORMAL_LATTICE_SECONDARY_RATIO_RANGE
            ),
            "lattice_cfa_mix": (
                cls.NORMAL_LATTICE_CFA_MIX_RANGE
            ),
            "lattice_achromatic_mix": (
                cls.NORMAL_LATTICE_ACHROMATIC_MIX_RANGE
            ),
            "lattice_chroma_boost": (
                cls.NORMAL_LATTICE_CHROMA_BOOST_RANGE
            ),
            "lattice_visibility_floor": (
                cls.NORMAL_LATTICE_VISIBILITY_FLOOR_RANGE
            ),
            "quadratic_carrier_fraction": (
                cls.NORMAL_QUADRATIC_CARRIER_FRACTION_RANGE
            ),
            "chromatic_dispersion": (
                cls.NORMAL_CHROMATIC_DISPERSION_RANGE
            ),
            "severe_strength_multiplier": (
                cls.NORMAL_SEVERE_STRENGTH_MULTIPLIER_RANGE
            ),
            "visibility_floor": cls.NORMAL_VISIBILITY_FLOOR_RANGE,
            "channel_response": cls.NORMAL_CHANNEL_RESPONSE_RANGE,
            "low_wave_interaction": (
                cls.NORMAL_LOW_WAVE_INTERACTION_RANGE
            ),
            "high_wave_interaction": (
                cls.NORMAL_HIGH_WAVE_INTERACTION_RANGE
            ),
            "radial_pixels": cls.NORMAL_RADIAL_PIXELS_RANGE,
            "bend_pixels": cls.NORMAL_BEND_PIXELS_RANGE,
            "ripple_pixels": cls.NORMAL_RIPPLE_PIXELS_RANGE,
            "perspective": cls.NORMAL_PERSPECTIVE_RANGE,
        }

    @staticmethod
    def _effective_beat_cycles_range(
        beat_cycles_range: Tuple[float, float],
        short_edge: float,
    ) -> Tuple[float, float]:
        # Keep at least four samples per visible beat on tiny test/crop sizes.
        # Normal training resolutions retain the configured range unchanged.
        maximum = min(beat_cycles_range[1], max(short_edge / 4.0, 0.25))
        minimum = min(beat_cycles_range[0], maximum)
        return minimum, maximum

    def _high_frequency_probability(self, is_extreme: bool) -> float:
        probability = self.high_frequency_probability
        if is_extreme and probability > 0.0:
            probability += self.EXTREME_HIGH_FREQUENCY_PROBABILITY_BOOST
        return min(probability, 1.0)

    def _secondary_order_probability(self, is_extreme: bool) -> float:
        probability = self.secondary_order_probability
        if is_extreme and probability > 0.0:
            probability += self.EXTREME_SECONDARY_ORDER_PROBABILITY_BOOST
        return min(probability, 1.0)

    @classmethod
    def _target_alias_period_profile(
        cls,
        is_extreme: bool,
    ) -> Tuple[Tuple[Tuple[float, float], ...], Tuple[float, ...]]:
        if is_extreme:
            return (
                cls.EXTREME_LATTICE_TARGET_ALIAS_PERIOD_BANDS,
                cls.EXTREME_LATTICE_SCALE_BAND_PROBABILITIES,
            )
        return (
            cls.NORMAL_LATTICE_TARGET_ALIAS_PERIOD_BANDS,
            cls.NORMAL_LATTICE_SCALE_BAND_PROBABILITIES,
        )

    def _sample_target_alias_period(
        self,
        reference: torch.Tensor,
        is_extreme: bool,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Draw one fine, medium, or coarse alias scale per batch sample."""

        batch = reference.shape[0]
        parameter_shape = (batch, 1, 1, 1)
        bands_value, probabilities = self._target_alias_period_profile(
            is_extreme
        )
        bands = reference.new_tensor(bands_value)
        selector = self._uniform(
            reference, 0.0, 1.0, parameter_shape, generator
        )
        first_threshold = probabilities[0]
        second_threshold = probabilities[0] + probabilities[1]
        band_index = (
            selector.ge(first_threshold).long()
            + selector.ge(second_threshold).long()
        )
        selected_bands = bands[band_index.view(-1)].view(batch, 1, 1, 1, 2)
        lower = selected_bands[..., 0]
        upper = selected_bands[..., 1]
        within_band = self._uniform(
            reference, 0.0, 1.0, parameter_shape, generator
        )
        period = torch.exp(
            torch.log(lower)
            + within_band * torch.log(upper / lower)
        )
        return period * self.alias_period_scale

    def _sample_optical_sigma(
        self,
        reference: torch.Tensor,
        is_extreme: bool,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        profile = self._profile_ranges(is_extreme)
        return self._uniform(
            reference,
            *profile["lattice_optical_sigma"],
            (reference.shape[0], 1, 1, 1),
            generator,
        )

    def _sample_display_layout_codes(
        self,
        reference: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        batch = reference.shape[0]
        if self.display_layout != "mixed":
            index = self.DISPLAY_LAYOUTS.index(self.display_layout)
            return torch.full(
                (batch,), index, device=reference.device, dtype=torch.long
            )
        selector = self._uniform(
            reference,
            0.0,
            1.0,
            (batch,),
            generator,
        )
        first = self.DISPLAY_LAYOUT_PROBABILITIES[0]
        second = first + self.DISPLAY_LAYOUT_PROBABILITIES[1]
        return selector.ge(first).long() + selector.ge(second).long()

    def _sample_cfa_codes(
        self,
        reference: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        batch = reference.shape[0]
        if self.sensor_cfa != "mixed":
            index = self.SENSOR_CFA_LAYOUTS.index(self.sensor_cfa)
            return torch.full(
                (batch,), index, device=reference.device, dtype=torch.long
            )
        selector = self._uniform(
            reference,
            0.0,
            1.0,
            (batch,),
            generator,
        )
        return torch.floor(selector * len(self.SENSOR_CFA_LAYOUTS)).long().clamp(
            max=len(self.SENSOR_CFA_LAYOUTS) - 1
        )

    def _display_layout_parameters(
        self,
        reference: torch.Tensor,
        layout_codes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return correlated order phases, amplitudes, and chroma gains."""

        base_phases = reference.new_tensor(self.DISPLAY_LAYOUT_BASE_PHASES)[
            layout_codes
        ]
        reciprocal_orders = reference.new_tensor(
            self.RECIPROCAL_LATTICE_ORDERS
        )
        order_phases = torch.einsum(
            "mu,buc->bmc", reciprocal_orders, base_phases
        )
        order_gains = reference.new_tensor(
            self.DISPLAY_LAYOUT_ORDER_GAINS
        )[layout_codes]
        chroma_gains = reference.new_tensor(
            self.DISPLAY_LAYOUT_CHROMA_GAINS
        )[layout_codes]
        return order_phases, order_gains, chroma_gains

    def _fit_correlated_sensor_carriers(
        self,
        reference: torch.Tensor,
        anchor_horizontal: torch.Tensor,
        anchor_vertical: torch.Tensor,
        sampled_alias_frequency: torch.Tensor,
        sampled_alias_angle: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fit one correlated ``u/v`` sensor carrier basis at the anchor.

        The previous implementation fitted all four reciprocal orders
        independently to one arbitrary target vector. Here only the two base
        lattice axes are fitted; diagonal carriers and target residuals are
        exact sums/differences. A log-domain blend with the native integer-fold
        residual prevents the requested scale bands from ignoring metadata
        geometry completely.
        """

        batch = reference.shape[0]
        base_gradients = torch.stack(
            (
                anchor_horizontal[:, :2],
                anchor_vertical[:, :2],
            ),
            dim=-1,
        )
        native_residual = base_gradients - base_gradients.round()
        if self.native_alias_blend == 1.0:
            # The exact endpoint must preserve the *vector*, including its
            # direction and DC case. Matching only its magnitude while using
            # a random angle did not implement the documented native fold.
            orders = reference.new_tensor(self.RECIPROCAL_LATTICE_ORDERS)
            carriers = torch.einsum("mu,buk->bmk", orders, base_gradients.round())
            residuals = torch.einsum("mu,buk->bmk", orders, native_residual)
            return carriers[..., 0], carriers[..., 1], residuals.square().sum(dim=-1).sqrt()
        native_frequency = native_residual.square().sum(dim=-1).sqrt()
        minimum_frequency = 1.0 / float(max(reference.shape[-2:]))
        native_frequency = native_frequency.clamp(
            min=minimum_frequency,
            max=0.49,
        )
        sampled_frequency = sampled_alias_frequency.view(batch, 1).expand(
            batch, 2
        )
        blend = self.native_alias_blend
        base_frequency = torch.exp(
            (1.0 - blend) * sampled_frequency.clamp_min(1e-6).log()
            + blend * native_frequency.log()
        )
        skew = self._uniform(
            reference,
            -math.pi / 12.0,
            math.pi / 12.0,
            (batch, 1),
            generator,
        )
        base_angle = sampled_alias_angle.view(batch, 1)
        axis_angles = torch.cat(
            (base_angle, base_angle + (math.pi / 2.0) + skew),
            dim=1,
        )
        desired_base_residual = base_frequency.unsqueeze(-1) * torch.stack(
            (torch.cos(axis_angles), torch.sin(axis_angles)),
            dim=-1,
        )
        base_carriers = base_gradients - desired_base_residual
        reciprocal_orders = reference.new_tensor(
            self.RECIPROCAL_LATTICE_ORDERS
        )
        mode_carriers = torch.einsum(
            "mu,buk->bmk", reciprocal_orders, base_carriers
        )
        desired_mode_residual = torch.einsum(
            "mu,buk->bmk", reciprocal_orders, desired_base_residual
        )
        desired_mode_frequency = desired_mode_residual.square().sum(
            dim=-1
        ).sqrt().clamp_min(1e-6)
        return (
            mode_carriers[..., 0],
            mode_carriers[..., 1],
            desired_mode_frequency,
        )

    def _sample_quadratic_carrier_coefficients(
        self,
        reference: torch.Tensor,
        target_alias_frequency: torch.Tensor,
        is_extreme: bool,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Sample correlated ``x^2, xy, y^2`` coefficients for four orders.

        Two base coefficient triplets describe the display ``u`` and ``v``
        axes. The diagonal reciprocal orders are exact sums/differences of
        those bases, preserving one two-dimensional lattice rather than drawing
        four unrelated phase warps. Coefficients are measured in cycles over
        normalized frame coordinates.
        """

        batch = reference.shape[0]
        profile = self._profile_ranges(is_extreme)
        raw_coefficients = self._uniform(
            reference,
            -1.0,
            1.0,
            (batch, 2, 3),
            generator,
        )
        coefficient_rms = raw_coefficients.square().mean(
            dim=-1,
            keepdim=True,
        ).sqrt().clamp_min(1e-6)
        normalized_coefficients = raw_coefficients / coefficient_rms
        carrier_fraction = self._uniform(
            reference,
            *profile["quadratic_carrier_fraction"],
            (batch, 2, 1),
            generator,
        )
        short_edge = float(min(reference.shape[-2:]))
        target_cycles = target_alias_frequency.view(batch, 1, 1) * short_edge
        base_coefficients = (
            normalized_coefficients
            * carrier_fraction
            * target_cycles
            * self.complexity_scale
            * self.quadratic_carrier_scale
        )
        reciprocal_orders = reference.new_tensor(
            self.RECIPROCAL_LATTICE_ORDERS
        )
        return torch.einsum(
            "mu,buk->bmk",
            reciprocal_orders,
            base_coefficients,
        )

    def _sample_chromatic_frequency_offsets(
        self,
        reference: torch.Tensor,
        is_extreme: bool,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Sample opposite, sub-percent R/B carrier offsets around green."""

        batch = reference.shape[0]
        profile = self._profile_ranges(is_extreme)
        parameter_shape = (batch, 1, 1, 1)
        magnitude = self._uniform(
            reference,
            *profile["chromatic_dispersion"],
            parameter_shape,
            generator,
        )
        direction = self._signed_uniform(
            reference,
            1.0,
            1.0,
            parameter_shape,
            generator,
        )
        balance = self._uniform(
            reference,
            *self.CHROMATIC_DISPERSION_BALANCE_RANGE,
            (batch, 2, 1, 1),
            generator,
        )
        effective_magnitude = (
            magnitude
            * self.complexity_scale
            * self.chromatic_dispersion_scale
        )
        red = direction * effective_magnitude * balance[:, 0:1]
        green = torch.zeros_like(red)
        blue = -direction * effective_magnitude * balance[:, 1:2]
        return torch.cat((red, green, blue), dim=1)

    def _severe_contrast_probability(self, is_extreme: bool) -> float:
        probability = self.severe_contrast_probability
        if is_extreme and probability > 0.0:
            probability += self.EXTREME_SEVERE_CONTRAST_PROBABILITY_BOOST
        return min(probability, 1.0)

    def _interference_probability(self, is_extreme: bool) -> float:
        probability = self.interference_probability
        if is_extreme and probability > 0.0:
            probability += self.EXTREME_INTERFERENCE_PROBABILITY_BOOST
        return min(probability, 1.0)

    def lattice_alias_period_range(
        self,
        height: int,
        width: int,
    ) -> Tuple[float, float]:
        """Estimate first-order local alias periods over the capture frame."""

        if height < 2 or width < 2:
            raise ValueError("height and width must both be at least 2")
        homography = (
            self._camera_to_screen_homography.detach().cpu().double()
        )
        screen_resolution = self._screen_resolution.detach().cpu().double()
        camera_resolution = self._camera_resolution.detach().cpu().double()
        effective_width = float(screen_resolution[0]) * (
            float(width - 1) / float(camera_resolution[0] - 1.0)
        )
        effective_height = float(screen_resolution[1]) * (
            float(height - 1) / float(camera_resolution[1] - 1.0)
        )
        modes = ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0))
        periods = []
        for y in (0.0, 0.25, 0.5, 0.75, 1.0):
            for x in (0.0, 0.25, 0.5, 0.75, 1.0):
                denominator = float(
                    homography[2, 0] * x
                    + homography[2, 1] * y
                    + homography[2, 2]
                )
                if abs(denominator) < 1e-8:
                    continue
                derivatives = []
                for row, effective_extent in (
                    (0, effective_width),
                    (1, effective_height),
                ):
                    numerator = float(
                        homography[row, 0] * x
                        + homography[row, 1] * y
                        + homography[row, 2]
                    )
                    derivative_x = float(
                        (
                            homography[row, 0] * denominator
                            - numerator * homography[2, 0]
                        )
                        / (denominator * denominator)
                    )
                    derivative_y = float(
                        (
                            homography[row, 1] * denominator
                            - numerator * homography[2, 1]
                        )
                        / (denominator * denominator)
                    )
                    derivatives.append(
                        (
                            derivative_x
                            * effective_extent
                            / float(width - 1),
                            derivative_y
                            * effective_extent
                            / float(height - 1),
                        )
                    )
                for horizontal_order, vertical_order in modes:
                    frequency_x = (
                        horizontal_order * derivatives[0][0]
                        + vertical_order * derivatives[1][0]
                    )
                    frequency_y = (
                        horizontal_order * derivatives[0][1]
                        + vertical_order * derivatives[1][1]
                    )
                    alias_x = frequency_x - round(frequency_x)
                    alias_y = frequency_y - round(frequency_y)
                    magnitude = math.hypot(alias_x, alias_y)
                    if magnitude > 1e-6 and math.isfinite(magnitude):
                        periods.append(1.0 / magnitude)
        if not periods:
            return float(max(height, width)), float(max(height, width))
        periods.sort()
        # Ignore a single near-Nyquist or near-DC outlier from an extrapolated
        # corner while retaining the genuine projective frequency spread.
        lower_index = int(0.05 * (len(periods) - 1))
        upper_index = int(0.95 * (len(periods) - 1))
        return periods[lower_index], min(
            periods[upper_index],
            float(max(height, width)),
        )

    def configuration_for(
        self,
        height: int,
        width: int,
        is_extreme: Optional[bool] = None,
    ) -> Dict[str, float]:
        """Return effective scalar ranges for an input resolution."""

        if height < 2 or width < 2:
            raise ValueError("height and width must both be at least 2")
        is_extreme = self._resolve_is_extreme(is_extreme)
        profile = self._profile_ranges(is_extreme)
        short_edge = float(min(height, width))
        phase_height = max(2, int(math.ceil(height * self.phase_scale)))
        phase_width = max(2, int(math.ceil(width * self.phase_scale)))
        phase_short_edge = float(min(phase_height, phase_width))
        beat_cycles = self._effective_beat_cycles_range(
            profile["beat_cycles"],
            phase_short_edge,
        )
        strength = self.modulation_strength_range or profile["strength"]
        exposure = profile["exposure"]
        high_probability = self._high_frequency_probability(is_extreme)
        secondary_order_probability = self._secondary_order_probability(
            is_extreme
        )
        severe_probability = self._severe_contrast_probability(
            is_extreme
        )
        interference_probability = self._interference_probability(is_extreme)
        legacy_active = float(self.legacy_low_frequency_weight > 0.0)
        severe_multiplier = profile["severe_strength_multiplier"]
        maximum_multiplier = (
            severe_multiplier[1] if severe_probability > 0.0 else 1.0
        )
        native_alias_period_min, native_alias_period_max = (
            self.lattice_alias_period_range(
                height,
                width,
            )
        )
        target_alias_period = tuple(
            value * self.alias_period_scale
            for value in profile["lattice_target_alias_period"]
        )
        scale_bands, scale_band_probabilities = (
            self._target_alias_period_profile(is_extreme)
        )
        lattice_visibility_floor = (
            profile["lattice_visibility_floor"]
            if self.lattice_visibility_floor is None
            else (
                self.lattice_visibility_floor,
                self.lattice_visibility_floor,
            )
        )
        effective_quadratic_scale = (
            self.complexity_scale * self.quadratic_carrier_scale
        )
        effective_dispersion_scale = (
            self.complexity_scale * self.chromatic_dispersion_scale
        )
        quadratic_fraction = tuple(
            value * effective_quadratic_scale
            for value in profile["quadratic_carrier_fraction"]
        )
        chromatic_dispersion = tuple(
            value * effective_dispersion_scale
            for value in profile["chromatic_dispersion"]
        )
        effective_white_balance_scale = (
            self.capture_lighting_scale * self.white_balance_scale
        )
        effective_glare_scale = (
            self.capture_lighting_scale * self.ambient_glare_scale
        )
        effective_tone_scale = (
            self.capture_lighting_scale * self.tone_curve_scale
        )
        white_balance_temperature = tuple(
            max(-0.35, min(0.35, value * effective_white_balance_scale))
            for value in profile["white_balance_temperature"]
        )
        white_balance_tint = tuple(
            max(-0.20, min(0.20, value * effective_white_balance_scale))
            for value in profile["white_balance_tint"]
        )
        ambient_glare = tuple(
            max(0.0, min(0.25, value * effective_glare_scale))
            for value in profile["ambient_glare"]
        )
        tone_gamma = tuple(
            max(
                0.65,
                min(1.50, 1.0 + (value - 1.0) * effective_tone_scale),
            )
            for value in profile["tone_gamma"]
        )
        return {
            "short_edge": short_edge,
            "phase_height": float(phase_height),
            "phase_width": float(phase_width),
            "phase_scale": self.phase_scale,
            "strength_min": strength[0] * self.strength_scale,
            "strength_max": (
                strength[1] * self.strength_scale * maximum_multiplier
            ),
            "base_strength_min": strength[0] * self.strength_scale,
            "base_strength_max": strength[1] * self.strength_scale,
            "global_coverage_probability": (
                1.0 if self.spatial_coverage == "global" else
                0.0 if self.spatial_coverage == "local" else
                self.global_coverage_probability
            ),
            "region_count_min": float(self.region_count_range[0]),
            "region_count_max": float(self.region_count_range[1]),
            "region_scale_min": self.region_scale_range[0],
            "region_scale_max": self.region_scale_range[1],
            "region_softness": self.region_softness,
            "severe_contrast_probability": severe_probability,
            "severe_strength_multiplier_min": severe_multiplier[0],
            "severe_strength_multiplier_max": severe_multiplier[1],
            "interference_probability": (
                interference_probability * legacy_active
            ),
            "interference_scale": self.interference_scale,
            "low_wave_interaction_min": (
                profile["low_wave_interaction"][0]
                * self.interference_scale
                * legacy_active
            ),
            "low_wave_interaction_max": (
                profile["low_wave_interaction"][1]
                * self.interference_scale
                * legacy_active
            ),
            "high_wave_interaction_min": (
                profile["high_wave_interaction"][0]
                * self.interference_scale
                * legacy_active
            ),
            "high_wave_interaction_max": (
                profile["high_wave_interaction"][1]
                * self.interference_scale
                * legacy_active
            ),
            "secondary_wave_weight_min": (
                self.SECONDARY_WEIGHT_RANGE[0] * legacy_active
            ),
            "secondary_wave_weight_max": (
                self.SECONDARY_WEIGHT_RANGE[1] * legacy_active
            ),
            "base_exposure_min": max(
                0.05,
                1.0 + self.exposure_scale * (exposure[0] - 1.0),
            ),
            "base_exposure_max": max(
                0.05,
                1.0 + self.exposure_scale * (exposure[1] - 1.0),
            ),
            "exposure_scale": self.exposure_scale,
            "vignette_strength_min": profile["vignette"][0],
            "vignette_strength_max": profile["vignette"][1],
            "illumination_gradient_min": profile[
                "illumination_gradient"
            ][0],
            "illumination_gradient_max": profile[
                "illumination_gradient"
            ][1],
            "beat_cycles_min": beat_cycles[0],
            "beat_cycles_max": beat_cycles[1],
            "beat_period_pixels_min": short_edge / beat_cycles[1],
            "beat_period_pixels_max": short_edge / beat_cycles[0],
            "high_frequency_probability": high_probability,
            "high_frequency_period_pixels_min": target_alias_period[0],
            "high_frequency_period_pixels_max": target_alias_period[1],
            "high_frequency_min_cycles_per_pixel": (
                1.0 / target_alias_period[1]
            ),
            "high_frequency_max_cycles_per_pixel": (
                1.0 / target_alias_period[0]
            ),
            "native_integer_fold_period_pixels_min": (
                native_alias_period_min
            ),
            "native_integer_fold_period_pixels_max": (
                native_alias_period_max
            ),
            "high_frequency_mix_min": profile["high_mix"][0],
            "high_frequency_mix_max": profile["high_mix"][1],
            "lattice_candidate_mode_count": float(
                len(self.RECIPROCAL_LATTICE_ORDERS)
            ),
            # Compatibility alias: these are candidates, not four simultaneous
            # screen-frequency layers.
            "lattice_mode_count": float(len(self.RECIPROCAL_LATTICE_ORDERS)),
            "lattice_simultaneous_mode_max": 2.0,
            "secondary_order_probability": secondary_order_probability,
            "secondary_order_ratio_min": profile[
                "lattice_secondary_ratio"
            ][0],
            "secondary_order_ratio_max": profile[
                "lattice_secondary_ratio"
            ][1],
            "target_alias_period_pixels_min": profile[
                "lattice_target_alias_period"
            ][0] * self.alias_period_scale,
            "target_alias_period_pixels_max": profile[
                "lattice_target_alias_period"
            ][1] * self.alias_period_scale,
            "alias_period_scale": self.alias_period_scale,
            "fine_alias_period_pixels_min": (
                scale_bands[0][0] * self.alias_period_scale
            ),
            "fine_alias_period_pixels_max": (
                scale_bands[0][1] * self.alias_period_scale
            ),
            "fine_alias_probability": scale_band_probabilities[0],
            "medium_alias_period_pixels_min": (
                scale_bands[1][0] * self.alias_period_scale
            ),
            "medium_alias_period_pixels_max": (
                scale_bands[1][1] * self.alias_period_scale
            ),
            "medium_alias_probability": scale_band_probabilities[1],
            "coarse_alias_period_pixels_min": (
                scale_bands[2][0] * self.alias_period_scale
            ),
            "coarse_alias_period_pixels_max": (
                scale_bands[2][1] * self.alias_period_scale
            ),
            "coarse_alias_probability": scale_band_probabilities[2],
            "lattice_pixel_fill_factor_min": profile[
                "lattice_pixel_fill_factor"
            ][0],
            "lattice_pixel_fill_factor_max": profile[
                "lattice_pixel_fill_factor"
            ][1],
            "cfa_mix_min": min(
                1.0,
                profile["lattice_cfa_mix"][0] * self.cfa_mix_scale,
            ),
            "cfa_mix_max": min(
                1.0,
                profile["lattice_cfa_mix"][1] * self.cfa_mix_scale,
            ),
            "achromatic_mix_min": profile["lattice_achromatic_mix"][0],
            "achromatic_mix_max": profile["lattice_achromatic_mix"][1],
            "chroma_boost_min": (
                profile["lattice_chroma_boost"][0] * self.chroma_scale
            ),
            "chroma_boost_max": (
                profile["lattice_chroma_boost"][1] * self.chroma_scale
            ),
            "chroma_scale": self.chroma_scale,
            "complexity_scale": self.complexity_scale,
            "quadratic_carrier_scale": self.quadratic_carrier_scale,
            "chromatic_dispersion_scale": self.chromatic_dispersion_scale,
            "quadratic_carrier_fraction_min": quadratic_fraction[0],
            "quadratic_carrier_fraction_max": quadratic_fraction[1],
            "quadratic_carrier_cycles_min": (
                quadratic_fraction[0]
                * short_edge
                / target_alias_period[1]
            ),
            "quadratic_carrier_cycles_max": (
                quadratic_fraction[1]
                * short_edge
                / target_alias_period[0]
            ),
            "chromatic_dispersion_min": (
                chromatic_dispersion[0]
                * self.CHROMATIC_DISPERSION_BALANCE_RANGE[0]
            ),
            "chromatic_dispersion_max": chromatic_dispersion[1],
            "maximum_red_blue_frequency_separation": (
                2.0 * chromatic_dispersion[1]
            ),
            "capture_lighting_scale": self.capture_lighting_scale,
            "white_balance_scale": self.white_balance_scale,
            "ambient_glare_scale": self.ambient_glare_scale,
            "tone_curve_scale": self.tone_curve_scale,
            "white_balance_temperature_min": white_balance_temperature[0],
            "white_balance_temperature_max": white_balance_temperature[1],
            "white_balance_tint_min": white_balance_tint[0],
            "white_balance_tint_max": white_balance_tint[1],
            "ambient_glare_min": ambient_glare[0],
            "ambient_glare_max": ambient_glare[1],
            "ambient_glare_sigma_min": profile["ambient_glare_sigma"][0],
            "ambient_glare_sigma_max": profile["ambient_glare_sigma"][1],
            "tone_gamma_min": tone_gamma[0],
            "tone_gamma_max": tone_gamma[1],
            "lattice_visibility_floor_min": lattice_visibility_floor[0],
            "lattice_visibility_floor_max": lattice_visibility_floor[1],
            "content_adaptive_strength": self.content_adaptive_strength,
            "content_adaptive_gain_min": (
                1.0
                - self.CONTENT_ADAPTIVE_NEUTRAL_SMOOTHNESS
                * self.content_adaptive_strength
            ),
            "content_adaptive_gain_max": (
                1.0
                + (1.0 - self.CONTENT_ADAPTIVE_NEUTRAL_SMOOTHNESS)
                * self.content_adaptive_strength
            ),
            "legacy_low_frequency_weight": self.legacy_low_frequency_weight,
            "geometry_jitter_scale": self.geometry_jitter_scale,
            "content_warp_scale": self.content_warp_scale,
            "content_warp_pixels_min": (
                profile["content_warp_pixels"][0] * self.content_warp_scale
            ),
            "content_warp_pixels_max": (
                profile["content_warp_pixels"][1] * self.content_warp_scale
            ),
            "native_alias_blend": self.native_alias_blend,
            "preserve_absolute_visibility": float(
                self.preserve_absolute_visibility
            ),
            "absolute_mtf_contrast_gain": (
                self.ABSOLUTE_MTF_CONTRAST_GAIN
                if self.preserve_absolute_visibility
                else 0.0
            ),
            "lattice_scale_jitter_min": (
                profile["lattice_scale_jitter"][0]
                * self.geometry_jitter_scale
            ),
            "lattice_scale_jitter_max": (
                profile["lattice_scale_jitter"][1]
                * self.geometry_jitter_scale
            ),
            "lattice_optical_sigma_min": profile[
                "lattice_optical_sigma"
            ][0],
            "lattice_optical_sigma_max": profile[
                "lattice_optical_sigma"
            ][1],
            "optical_psf_scale": self.optical_psf_scale,
            "image_psf_sigma_pixels_min": (
                profile["lattice_optical_sigma"][0]
                * self.LATTICE_SIGMA_TO_IMAGE_PSF
                * self.optical_psf_scale
            ),
            "image_psf_sigma_pixels_max": (
                profile["lattice_optical_sigma"][1]
                * self.LATTICE_SIGMA_TO_IMAGE_PSF
                * self.optical_psf_scale
            ),
            "sensor_isp_scale": self.sensor_isp_scale,
            "full_signal_cfa_scale": self.full_signal_cfa_scale,
            "full_signal_cfa_mix_min": min(
                1.0,
                profile["full_signal_cfa_mix"][0]
                * self.sensor_isp_scale
                * self.full_signal_cfa_scale,
            ),
            "full_signal_cfa_mix_max": min(
                1.0,
                profile["full_signal_cfa_mix"][1]
                * self.sensor_isp_scale
                * self.full_signal_cfa_scale,
            ),
            "sensor_noise_scale": self.sensor_noise_scale,
            "sensor_shot_variance_min": (
                profile["sensor_shot_variance"][0]
                * (self.sensor_isp_scale * self.sensor_noise_scale) ** 2
            ),
            "sensor_shot_variance_max": (
                profile["sensor_shot_variance"][1]
                * (self.sensor_isp_scale * self.sensor_noise_scale) ** 2
            ),
            "sensor_read_std_min": (
                profile["sensor_read_std"][0]
                * self.sensor_isp_scale
                * self.sensor_noise_scale
            ),
            "sensor_read_std_max": (
                profile["sensor_read_std"][1]
                * self.sensor_isp_scale
                * self.sensor_noise_scale
            ),
            "color_correction_scale": self.color_correction_scale,
            "color_matrix_jitter_min": (
                profile["color_matrix_jitter"][0]
                * self.sensor_isp_scale
                * self.color_correction_scale
            ),
            "color_matrix_jitter_max": (
                profile["color_matrix_jitter"][1]
                * self.sensor_isp_scale
                * self.color_correction_scale
            ),
            "tone_s_curve_min": (
                profile["tone_s_curve"][0]
                * self.sensor_isp_scale
                * self.color_correction_scale
            ),
            "tone_s_curve_max": (
                profile["tone_s_curve"][1]
                * self.sensor_isp_scale
                * self.color_correction_scale
            ),
            "display_layout_profile_count": float(
                len(self.DISPLAY_LAYOUTS)
            ),
            "sensor_cfa_profile_count": float(
                len(self.SENSOR_CFA_LAYOUTS)
            ),
            "capture_screen_width_px": float(self._screen_resolution[0]),
            "capture_screen_height_px": float(self._screen_resolution[1]),
            "capture_camera_width_px": float(self._camera_resolution[0]),
            "capture_camera_height_px": float(self._camera_resolution[1]),
            "visibility_floor_min": profile["visibility_floor"][0],
            "visibility_floor_max": profile["visibility_floor"][1],
            "channel_response_min": profile["channel_response"][0],
            "channel_response_max": profile["channel_response"][1],
            "sensor_frequency_min_cycles_per_pixel": (
                self.SENSOR_FREQUENCY_RANGE[0]
            ),
            "sensor_frequency_max_cycles_per_pixel": (
                self.SENSOR_FREQUENCY_RANGE[1]
            ),
            "radial_displacement_pixels_min": profile["radial_pixels"][0],
            "radial_displacement_pixels_max": profile["radial_pixels"][1],
            "bend_displacement_pixels_min": profile["bend_pixels"][0],
            "bend_displacement_pixels_max": profile["bend_pixels"][1],
            "ripple_displacement_pixels_min": profile["ripple_pixels"][0],
            "ripple_displacement_pixels_max": profile["ripple_pixels"][1],
        }

    def generate_illumination(
        self,
        reference: torch.Tensor,
        is_extreme: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Generate linear exposure loss, off-centre vignette, and gradient."""

        batch, _, height, width = reference.shape
        if self.exposure_scale == 0.0:
            return reference.new_ones((batch, 1, height, width))
        profile = self._profile_ranges(is_extreme)
        parameter_shape = (batch, 1, 1, 1)
        _, _, normalized_x, normalized_y = self._pixel_grid(
            reference,
            height,
            width,
            height,
            width,
        )

        base_exposure = self._uniform(
            reference,
            *profile["exposure"],
            parameter_shape,
            generator,
        )
        centre_x = self._uniform(
            reference,
            -0.25,
            0.25,
            parameter_shape,
            generator,
        )
        centre_y = self._uniform(
            reference,
            -0.25,
            0.25,
            parameter_shape,
            generator,
        )
        radius_squared = 0.5 * (
            (normalized_x - centre_x).square()
            + (normalized_y - centre_y).square()
        )
        vignette_strength = self._uniform(
            reference,
            *profile["vignette"],
            parameter_shape,
            generator,
        )
        vignette = 1.0 - vignette_strength * radius_squared.clamp_max(1.5)

        gradient_strength = self._uniform(
            reference,
            *profile["illumination_gradient"],
            parameter_shape,
            generator,
        )
        gradient_theta = self._uniform(
            reference,
            0.0,
            2.0 * math.pi,
            parameter_shape,
            generator,
        )
        gradient_projection = 0.5 * (
            torch.cos(gradient_theta) * normalized_x
            + torch.sin(gradient_theta) * normalized_y
        )
        field = base_exposure * vignette * (
            1.0 + gradient_strength * gradient_projection
        )
        field = 1.0 + self.exposure_scale * (field - 1.0)
        return field.clamp(0.05, 1.0)

    @staticmethod
    def _normalized_colour_gains(
        temperature: torch.Tensor,
        tint: torch.Tensor,
    ) -> torch.Tensor:
        """Convert log-temperature/tint shifts to unit-luminance RGB gains."""

        gains = torch.cat(
            (
                torch.exp(temperature),
                torch.exp(tint),
                torch.exp(-temperature),
            ),
            dim=1,
        )
        luminance_weights = gains.new_tensor(
            (0.2126, 0.7152, 0.0722)
        ).view(1, 3, 1, 1)
        luminance_gain = (gains * luminance_weights).sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        return gains / luminance_gain

    def generate_capture_lighting(
        self,
        reference: torch.Tensor,
        is_extreme: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate white balance, linear ambient veil, and output gamma.

        The broad additive veil approximates ambient screen reflection and lens
        flare after the display exposure loss. It intentionally lifts black
        levels slightly, unlike the multiplicative illumination field.
        """

        batch, _, height, width = reference.shape
        profile = self._profile_ranges(is_extreme)
        parameter_shape = (batch, 1, 1, 1)
        white_balance_strength = (
            self.capture_lighting_scale * self.white_balance_scale
        )
        glare_strength_scale = (
            self.capture_lighting_scale * self.ambient_glare_scale
        )
        tone_strength = self.capture_lighting_scale * self.tone_curve_scale

        temperature = (
            self._uniform(
                reference,
                *profile["white_balance_temperature"],
                parameter_shape,
                generator,
            )
            * white_balance_strength
        ).clamp(-0.35, 0.35)
        tint = (
            self._uniform(
                reference,
                *profile["white_balance_tint"],
                parameter_shape,
                generator,
            )
            * white_balance_strength
        ).clamp(-0.20, 0.20)
        white_balance = self._normalized_colour_gains(temperature, tint)

        _, _, normalized_x, normalized_y = self._pixel_grid(
            reference,
            height,
            width,
            height,
            width,
        )
        centre_x = self._uniform(
            reference, -0.65, 0.65, parameter_shape, generator
        )
        centre_y = self._uniform(
            reference, -0.65, 0.65, parameter_shape, generator
        )
        sigma_x = self._uniform(
            reference,
            *profile["ambient_glare_sigma"],
            parameter_shape,
            generator,
        )
        sigma_y = self._uniform(
            reference,
            *profile["ambient_glare_sigma"],
            parameter_shape,
            generator,
        )
        glare_angle = self._uniform(
            reference,
            0.0,
            2.0 * math.pi,
            parameter_shape,
            generator,
        )
        delta_x = normalized_x - centre_x
        delta_y = normalized_y - centre_y
        rotated_x = (
            torch.cos(glare_angle) * delta_x
            + torch.sin(glare_angle) * delta_y
        )
        rotated_y = (
            -torch.sin(glare_angle) * delta_x
            + torch.cos(glare_angle) * delta_y
        )
        glare_lobe = torch.exp(
            -0.5
            * (
                (rotated_x / sigma_x).square()
                + (rotated_y / sigma_y).square()
            )
        )
        glare_floor = self._uniform(
            reference,
            *self.AMBIENT_GLARE_FLOOR_RANGE,
            parameter_shape,
            generator,
        )
        glare_envelope = glare_floor + (1.0 - glare_floor) * glare_lobe
        glare_strength = (
            self._uniform(
                reference,
                *profile["ambient_glare"],
                parameter_shape,
                generator,
            )
            * glare_strength_scale
        ).clamp(0.0, 0.25)
        glare_temperature = (
            self._uniform(
                reference,
                *profile["white_balance_temperature"],
                parameter_shape,
                generator,
            )
            * glare_strength_scale
        ).clamp(-0.35, 0.35)
        glare_tint = (
            self._uniform(
                reference,
                *profile["white_balance_tint"],
                parameter_shape,
                generator,
            )
            * glare_strength_scale
        ).clamp(-0.20, 0.20)
        glare_colour = self._normalized_colour_gains(
            glare_temperature,
            glare_tint,
        )
        ambient_veil = glare_strength * glare_envelope * glare_colour

        sampled_gamma = self._uniform(
            reference,
            *profile["tone_gamma"],
            parameter_shape,
            generator,
        )
        tone_gamma = (
            1.0 + (sampled_gamma - 1.0) * tone_strength
        ).clamp(0.65, 1.50)
        return white_balance, ambient_veil, tone_gamma

    def _apply_capture_lighting(
        self,
        image: torch.Tensor,
        is_extreme: bool,
        generator: Optional[torch.Generator],
        optical_sigma: Optional[torch.Tensor] = None,
        cfa_codes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compatibility wrapper around the linear capture pipeline."""

        if optical_sigma is None:
            optical_sigma = self._sample_optical_sigma(
                image,
                is_extreme=is_extreme,
                generator=generator,
            )
        if cfa_codes is None:
            cfa_codes = self._sample_cfa_codes(image, generator)
        return self._apply_linear_capture_pipeline(
            self._srgb_to_linear(image.clamp(0.0, 1.0)),
            is_extreme=is_extreme,
            generator=generator,
            optical_sigma=optical_sigma,
            cfa_codes=cfa_codes,
        )

    def projected_lattice_coordinates(
        self,
        reference: torch.Tensor,
        is_extreme: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map camera pixels to metadata-calibrated display-pixel coordinates.

        Sampling these coordinates at integer camera pixels performs the
        Nyquist folding implicitly. Small per-sample perturbations model nearby
        camera poses while preserving the calibrated projective structure.
        """

        batch, _, height, width = reference.shape
        parameter_shape = (batch, 1, 1, 1)
        profile = self._profile_ranges(is_extreme)
        _, _, normalized_x, normalized_y = self._pixel_grid(
            reference,
            height,
            width,
            height,
            width,
        )
        camera_x = 0.5 * normalized_x
        camera_y = 0.5 * normalized_y

        jitter_scale = self.geometry_jitter_scale
        roll_degrees = self._uniform(
            reference,
            *profile["lattice_roll_jitter_degrees"],
            parameter_shape,
            generator,
        ) * jitter_scale
        roll = roll_degrees * (math.pi / 180.0)
        cos_roll = torch.cos(roll)
        sin_roll = torch.sin(roll)
        rotated_x = cos_roll * camera_x - sin_roll * camera_y
        rotated_y = sin_roll * camera_x + cos_roll * camera_y

        scale_x = 1.0 + self._uniform(
            reference,
            *profile["lattice_scale_jitter"],
            parameter_shape,
            generator,
        ) * jitter_scale
        scale_y = 1.0 + self._uniform(
            reference,
            *profile["lattice_scale_jitter"],
            parameter_shape,
            generator,
        ) * jitter_scale
        scale_x = scale_x.clamp_min(0.5)
        scale_y = scale_y.clamp_min(0.5)
        translation_x = self._uniform(
            reference,
            *profile["lattice_translation_jitter"],
            parameter_shape,
            generator,
        ) * jitter_scale
        translation_y = self._uniform(
            reference,
            *profile["lattice_translation_jitter"],
            parameter_shape,
            generator,
        ) * jitter_scale
        camera_x = 0.5 + scale_x * rotated_x + translation_x
        camera_y = 0.5 + scale_y * rotated_y + translation_y

        homography = self._camera_to_screen_homography.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        denominator = (
            homography[2, 0] * camera_x
            + homography[2, 1] * camera_y
            + homography[2, 2]
        )
        denominator_sign = torch.where(
            denominator >= 0.0,
            torch.ones_like(denominator),
            -torch.ones_like(denominator),
        )
        denominator = torch.where(
            denominator.abs() < 1e-6,
            denominator_sign * 1e-6,
            denominator,
        )
        screen_u = (
            homography[0, 0] * camera_x
            + homography[0, 1] * camera_y
            + homography[0, 2]
        ) / denominator
        screen_v = (
            homography[1, 0] * camera_x
            + homography[1, 1] * camera_y
            + homography[1, 2]
        ) / denominator

        screen_resolution = self._screen_resolution.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        camera_resolution = self._camera_resolution.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        effective_screen_width = screen_resolution[0] * (
            float(width - 1) / (camera_resolution[0] - 1.0)
        )
        effective_screen_height = screen_resolution[1] * (
            float(height - 1) / (camera_resolution[1] - 1.0)
        )
        return (
            screen_u * effective_screen_width,
            screen_v * effective_screen_height,
        )

    def _warp_content_with_projected_residual(
        self,
        image: torch.Tensor,
        projected_coordinates: Tuple[torch.Tensor, torch.Tensor],
        is_extreme: bool,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Apply a bounded residual warp from the same lattice homography.

        Screen-camera watermark pipelines commonly rectify the captured screen
        before decoding. We therefore remove the local affine component of the
        metadata mapping and retain only a small projective residual instead of
        rendering the full camera frame with large outside-screen borders.
        """

        if self.content_warp_scale == 0.0:
            return image
        screen_x, screen_y = projected_coordinates
        batch, _, height, width = image.shape
        centre_y = height // 2
        centre_x = width // 2
        horizontal_x, vertical_x = self._local_spatial_gradients(screen_x)
        horizontal_y, vertical_y = self._local_spatial_gradients(screen_y)

        pixel_x = torch.arange(
            width, device=image.device, dtype=image.dtype
        ).view(1, 1, 1, width)
        pixel_y = torch.arange(
            height, device=image.device, dtype=image.dtype
        ).view(1, 1, height, 1)
        relative_x = pixel_x - float(centre_x)
        relative_y = pixel_y - float(centre_y)

        def projective_residual(
            coordinate: torch.Tensor,
            horizontal: torch.Tensor,
            vertical: torch.Tensor,
        ) -> torch.Tensor:
            centre = coordinate[:, :, centre_y, centre_x].view(
                batch, 1, 1, 1
            )
            slope_x = horizontal[:, :, centre_y, centre_x].view(
                batch, 1, 1, 1
            )
            slope_y = vertical[:, :, centre_y, centre_x].view(
                batch, 1, 1, 1
            )
            affine = centre + slope_x * relative_x + slope_y * relative_y
            return coordinate - affine

        residual_x = projective_residual(
            screen_x, horizontal_x, vertical_x
        )
        residual_y = projective_residual(
            screen_y, horizontal_y, vertical_y
        )
        residual_peak = (
            residual_x.square() + residual_y.square()
        ).sqrt().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        profile = self._profile_ranges(is_extreme)
        target_pixels = self._uniform(
            image,
            *profile["content_warp_pixels"],
            (batch, 1, 1, 1),
            generator,
        ) * self.content_warp_scale
        displacement_x = residual_x * (target_pixels / residual_peak)
        displacement_y = residual_y * (target_pixels / residual_peak)

        identity_x = torch.linspace(
            -1.0, 1.0, width, device=image.device, dtype=image.dtype
        ).view(1, 1, width).expand(batch, height, width)
        identity_y = torch.linspace(
            -1.0, 1.0, height, device=image.device, dtype=image.dtype
        ).view(1, height, 1).expand(batch, height, width)
        grid_x = identity_x + (
            2.0 * displacement_x[:, 0] / float(max(width - 1, 1))
        )
        grid_y = identity_y + (
            2.0 * displacement_y[:, 0] / float(max(height - 1, 1))
        )
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    @staticmethod
    def _local_spatial_gradients(
        coordinate: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        horizontal = coordinate[:, :, :, 1:] - coordinate[:, :, :, :-1]
        vertical = coordinate[:, :, 1:, :] - coordinate[:, :, :-1, :]
        horizontal = F.pad(horizontal, (0, 1, 0, 0), mode="replicate")
        vertical = F.pad(vertical, (0, 0, 0, 1), mode="replicate")
        return horizontal, vertical

    @classmethod
    def _local_spatial_frequency(cls, coordinate: torch.Tensor) -> torch.Tensor:
        horizontal, vertical = cls._local_spatial_gradients(coordinate)
        return (horizontal.square() + vertical.square()).sqrt()

    def _bayer_demosaic_pattern(
        self,
        pattern: torch.Tensor,
        generator: Optional[torch.Generator],
        cfa_codes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply a lightweight Bayer sample/demosaic approximation."""

        batch, _, height, width = pattern.shape
        row = torch.arange(
            height, device=pattern.device, dtype=pattern.dtype
        ).view(1, 1, height, 1)
        column = torch.arange(
            width, device=pattern.device, dtype=pattern.dtype
        ).view(1, 1, 1, width)
        even_row = torch.remainder(row, 2.0).eq(0.0)
        even_column = torch.remainder(column, 2.0).eq(0.0)
        even_even = (even_row & even_column).to(pattern.dtype)
        odd_odd = ((~even_row) & (~even_column)).to(pattern.dtype)
        even_odd = (even_row & (~even_column)).to(pattern.dtype)
        odd_even = ((~even_row) & even_column).to(pattern.dtype)

        def masks_for(
            red_mask: torch.Tensor,
            blue_mask: torch.Tensor,
        ) -> torch.Tensor:
            green_mask = 1.0 - red_mask - blue_mask
            return torch.cat((red_mask, green_mask, blue_mask), dim=1)

        layout_masks = torch.stack(
            (
                masks_for(even_even, odd_odd),
                masks_for(odd_odd, even_even),
                masks_for(even_odd, odd_even),
                masks_for(odd_even, even_odd),
            ),
            dim=0,
        )[:, 0]
        if cfa_codes is None:
            cfa_codes = self._sample_cfa_codes(pattern, generator)
        masks = layout_masks[cfa_codes]

        raw = (pattern * masks).sum(dim=1, keepdim=True)
        sparse = raw * masks
        kernel = pattern.new_tensor(
            ((1.0, 2.0, 1.0), (2.0, 4.0, 2.0), (1.0, 2.0, 1.0))
        ).view(1, 1, 3, 3)
        sparse_flat = sparse.reshape(batch * 3, 1, height, width)
        mask_flat = masks.reshape(batch * 3, 1, height, width)
        numerator = F.conv2d(sparse_flat, kernel, padding=1)
        denominator = F.conv2d(mask_flat, kernel, padding=1).clamp_min(1e-6)
        return (numerator / denominator).reshape(batch, 3, height, width)

    def _apply_optical_psf(
        self,
        image: torch.Tensor,
        optical_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one separable per-sample Gaussian PSF to full radiance."""

        if self.optical_psf_scale == 0.0:
            return image
        batch, channels, height, width = image.shape
        sigma_pixels = (
            optical_sigma.view(batch)
            * self.LATTICE_SIGMA_TO_IMAGE_PSF
            * self.optical_psf_scale
        ).clamp_min(0.05)
        radius = 3
        offsets = torch.arange(
            -radius,
            radius + 1,
            device=image.device,
            dtype=image.dtype,
        ).view(1, -1)
        kernels = torch.exp(
            -0.5 * (offsets / sigma_pixels.view(batch, 1)).square()
        )
        kernels = kernels / kernels.sum(dim=1, keepdim=True).clamp_min(1e-8)
        kernels = kernels.repeat_interleave(channels, dim=0)
        flattened = image.reshape(1, batch * channels, height, width)
        horizontal_kernel = kernels.view(batch * channels, 1, 1, -1)
        padded = F.pad(flattened, (radius, radius, 0, 0), mode="replicate")
        filtered = F.conv2d(
            padded,
            horizontal_kernel,
            groups=batch * channels,
        )
        vertical_kernel = kernels.view(batch * channels, 1, -1, 1)
        padded = F.pad(filtered, (0, 0, radius, radius), mode="replicate")
        filtered = F.conv2d(
            padded,
            vertical_kernel,
            groups=batch * channels,
        )
        return filtered.reshape(batch, channels, height, width)

    def _apply_full_signal_cfa(
        self,
        image: torch.Tensor,
        is_extreme: bool,
        generator: Optional[torch.Generator],
        cfa_codes: torch.Tensor,
    ) -> torch.Tensor:
        effective_scale = self.sensor_isp_scale * self.full_signal_cfa_scale
        if effective_scale == 0.0:
            return image
        profile = self._profile_ranges(is_extreme)
        mix = self._uniform(
            image,
            *profile["full_signal_cfa_mix"],
            (image.shape[0], 1, 1, 1),
            generator,
        )
        mix = (mix * effective_scale).clamp(0.0, 1.0)
        demosaiced = self._bayer_demosaic_pattern(
            image,
            generator,
            cfa_codes=cfa_codes,
        )
        return image + mix * (demosaiced - image)

    def _apply_sensor_noise(
        self,
        image: torch.Tensor,
        is_extreme: bool,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        effective_scale = self.sensor_isp_scale * self.sensor_noise_scale
        if effective_scale == 0.0:
            return image
        profile = self._profile_ranges(is_extreme)
        parameter_shape = (image.shape[0], 1, 1, 1)
        shot_variance = self._uniform(
            image,
            *profile["sensor_shot_variance"],
            parameter_shape,
            generator,
        )
        read_std = self._uniform(
            image,
            *profile["sensor_read_std"],
            parameter_shape,
            generator,
        )
        standard_deviation = (
            shot_variance * image.clamp_min(0.0) + read_std.square()
        ).sqrt() * effective_scale
        noise = torch.randn(
            image.shape,
            device=image.device,
            dtype=image.dtype,
            generator=generator,
        )
        return image + standard_deviation * noise

    def _apply_colour_correction_and_tone(
        self,
        image: torch.Tensor,
        white_balance: torch.Tensor,
        is_extreme: bool,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        balanced = image * white_balance
        effective_scale = self.sensor_isp_scale * self.color_correction_scale
        if effective_scale == 0.0:
            return balanced
        batch = image.shape[0]
        profile = self._profile_ranges(is_extreme)
        amplitude = self._uniform(
            image,
            *profile["color_matrix_jitter"],
            (batch, 1, 1),
            generator,
        ) * effective_scale
        jitter = self._uniform(
            image,
            -1.0,
            1.0,
            (batch, 3, 3),
            generator,
        )
        # A zero row sum keeps neutral gray neutral while permitting realistic
        # cross-channel colour correction on saturated content.
        jitter = jitter - jitter.mean(dim=-1, keepdim=True)
        identity = torch.eye(
            3, device=image.device, dtype=image.dtype
        ).view(1, 3, 3)
        matrix = identity + amplitude * jitter
        corrected = torch.einsum("bij,bjhw->bihw", matrix, balanced)
        corrected = corrected.clamp(0.0, 1.0)
        tone_strength = self._uniform(
            image,
            *profile["tone_s_curve"],
            (batch, 1, 1, 1),
            generator,
        )
        tone_strength = (tone_strength * effective_scale).clamp(0.0, 0.45)
        smoothstep = corrected.square() * (3.0 - 2.0 * corrected)
        return corrected + tone_strength * (smoothstep - corrected)

    def _apply_linear_capture_pipeline(
        self,
        linear_image: torch.Tensor,
        is_extreme: bool,
        generator: Optional[torch.Generator],
        optical_sigma: torch.Tensor,
        cfa_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Apply optics, ambient radiance, sensor sampling, and output tone."""

        radiance = self._apply_optical_psf(
            linear_image.clamp(0.0, 1.0),
            optical_sigma,
        )
        white_balance, ambient_veil, tone_gamma = (
            self.generate_capture_lighting(
                radiance,
                is_extreme=is_extreme,
                generator=generator,
            )
        )
        radiance = radiance + ambient_veil * (1.0 - radiance)
        radiance = self._apply_full_signal_cfa(
            radiance,
            is_extreme=is_extreme,
            generator=generator,
            cfa_codes=cfa_codes,
        )
        radiance = self._apply_sensor_noise(
            radiance,
            is_extreme=is_extreme,
            generator=generator,
        )
        radiance = self._apply_colour_correction_and_tone(
            radiance,
            white_balance,
            is_extreme=is_extreme,
            generator=generator,
        ).clamp(0.0, 1.0)
        rendered = self._linear_to_srgb(radiance)
        tone_mapped = rendered.clamp_min(1e-8).pow(tone_gamma)
        return torch.where(rendered > 0.0, tone_mapped, rendered)

    def generate_high_frequency_moire(
        self,
        reference: torch.Tensor,
        is_extreme: bool = False,
        generator: Optional[torch.Generator] = None,
        projected_coordinates: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        optical_sigma: Optional[torch.Tensor] = None,
        display_layout_codes: Optional[torch.Tensor] = None,
        cfa_codes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate a dominant-order projective display/CFA lattice alias.

        A real screen contributes many reciprocal-lattice orders, but the lens
        MTF, sensor pixel aperture, and demosaicing normally leave one order
        dominant at a given capture geometry. We therefore score four candidate
        orders physically, choose one dominant order per sample, and admit at
        most one weak second order. A correlated quadratic carrier adds local
        chirp while preserving the same lattice, and anchor-relative RGB
        frequency offsets introduce gradual colour drift. Spatial visibility
        changes smoothly with the projective Jacobian, so one selected order
        can naturally transition from fine fringes to broad curved beats across
        the frame.
        """

        batch = reference.shape[0]
        mode_count = len(self.RECIPROCAL_LATTICE_ORDERS)
        parameter_shape = (batch, 1, 1, 1)
        profile = self._profile_ranges(is_extreme)
        if projected_coordinates is None:
            projected_coordinates = self.projected_lattice_coordinates(
                reference,
                is_extreme=is_extreme,
                generator=generator,
            )
        screen_x, screen_y = projected_coordinates
        coordinates = torch.cat(
            (screen_x, screen_y, screen_x + screen_y, screen_x - screen_y),
            dim=1,
        )

        horizontal, vertical = self._local_spatial_gradients(coordinates)
        raw_frequency = (
            horizontal.square() + vertical.square()
        ).sqrt().clamp_max(8.0)
        if optical_sigma is None:
            optical_sigma = self._sample_optical_sigma(
                reference,
                is_extreme=is_extreme,
                generator=generator,
            )
        lens_mtf = torch.exp(
            -2.0
            * math.pi
            * math.pi
            * optical_sigma.square()
            * raw_frequency.square()
        )
        pixel_fill_factor = self._uniform(
            reference,
            *profile["lattice_pixel_fill_factor"],
            parameter_shape,
            generator,
        )
        # Pixel integration is a rectangular aperture. Its separable sinc
        # response suppresses reciprocal orders near aperture zeros instead of
        # preserving every order with an artificial non-zero MTF floor.
        aperture_mtf = (
            torch.sinc(pixel_fill_factor * horizontal)
            * torch.sinc(pixel_fill_factor * vertical)
        ).abs()
        optical_visibility = lens_mtf * aperture_mtf
        if display_layout_codes is None:
            display_layout_codes = self._sample_display_layout_codes(
                reference, generator
            )
        (
            layout_order_phases,
            layout_order_gains,
            layout_chroma_gains,
        ) = self._display_layout_parameters(
            reference,
            display_layout_codes,
        )
        optical_visibility = optical_visibility * layout_order_gains.view(
            batch, mode_count, 1, 1
        )

        target_period = self._sample_target_alias_period(
            reference,
            is_extreme=is_extreme,
            generator=generator,
        )
        target_alias_frequency = target_period.reciprocal()
        target_alias_angle = self._uniform(
            reference,
            0.0,
            2.0 * math.pi,
            parameter_shape,
            generator,
        )

        # The metadata describes the rendered output rather than the camera's
        # raw sensor lattice. JPEG resizing and the ISP introduce an unknown
        # affine sampling carrier. Fit one correlated u/v carrier basis at a
        # random anchor, derive diagonal carriers by exact sums/differences,
        # and retain the projective residual away from that anchor.
        anchor_x_unit = self._uniform(
            reference, 0.15, 0.85, parameter_shape, generator
        )
        anchor_y_unit = self._uniform(
            reference, 0.15, 0.85, parameter_shape, generator
        )
        height, width = reference.shape[-2:]
        anchor_x = (anchor_x_unit.view(batch) * float(width - 1)).round().long()
        anchor_y = (anchor_y_unit.view(batch) * float(height - 1)).round().long()
        anchor_flat = (anchor_y * width + anchor_x).view(batch, 1, 1)
        anchor_flat = anchor_flat.expand(batch, mode_count, 1)
        anchor_horizontal = torch.gather(
            horizontal.flatten(2), 2, anchor_flat
        ).squeeze(-1)
        anchor_vertical = torch.gather(
            vertical.flatten(2), 2, anchor_flat
        ).squeeze(-1)
        (
            sensor_carrier_horizontal,
            sensor_carrier_vertical,
            desired_mode_frequency,
        ) = self._fit_correlated_sensor_carriers(
            reference,
            anchor_horizontal,
            anchor_vertical,
            target_alias_frequency,
            target_alias_angle,
            generator,
        )
        pixel_x = torch.arange(
            width, device=reference.device, dtype=reference.dtype
        ).view(1, 1, 1, width)
        pixel_y = torch.arange(
            height, device=reference.device, dtype=reference.dtype
        ).view(1, 1, height, 1)
        relative_x = pixel_x - anchor_x.to(reference.dtype).view(batch, 1, 1, 1)
        relative_y = pixel_y - anchor_y.to(reference.dtype).view(batch, 1, 1, 1)
        sensor_carrier_plane = (
            sensor_carrier_horizontal.view(batch, mode_count, 1, 1)
            * relative_x
            + sensor_carrier_vertical.view(batch, mode_count, 1, 1)
            * relative_y
        )
        relative_x_normalized = relative_x / float(max(width - 1, 1))
        relative_y_normalized = relative_y / float(max(height - 1, 1))
        quadratic_coefficients = (
            self._sample_quadratic_carrier_coefficients(
                reference,
                target_alias_frequency,
                is_extreme=is_extreme,
                generator=generator,
            )
        )
        q_xx = quadratic_coefficients[:, :, 0].view(
            batch, mode_count, 1, 1
        )
        q_xy = quadratic_coefficients[:, :, 1].view(
            batch, mode_count, 1, 1
        )
        q_yy = quadratic_coefficients[:, :, 2].view(
            batch, mode_count, 1, 1
        )
        quadratic_carrier = (
            q_xx * relative_x_normalized.square()
            + q_xy * relative_x_normalized * relative_y_normalized
            + q_yy * relative_y_normalized.square()
        )
        quadratic_horizontal = (
            2.0 * q_xx * relative_x_normalized
            + q_xy * relative_y_normalized
        ) / float(max(width - 1, 1))
        quadratic_vertical = (
            q_xy * relative_x_normalized
            + 2.0 * q_yy * relative_y_normalized
        ) / float(max(height - 1, 1))
        alias_horizontal = (
            horizontal
            - sensor_carrier_horizontal.view(batch, mode_count, 1, 1)
            - quadratic_horizontal
        )
        alias_vertical = (
            vertical
            - sensor_carrier_vertical.view(batch, mode_count, 1, 1)
            - quadratic_vertical
        )
        alias_frequency = (
            alias_horizontal.square() + alias_vertical.square()
        ).sqrt()
        beat_coordinates = (
            coordinates - sensor_carrier_plane - quadratic_carrier
        )
        log_frequency_error = torch.log(
            alias_frequency.clamp_min(1e-6)
            / desired_mode_frequency.view(batch, mode_count, 1, 1)
        )
        alias_preference = torch.exp(
            -0.5
            * (
                log_frequency_error / self.LATTICE_ALIAS_LOG_BANDWIDTH
            ).square()
        )
        visibility = optical_visibility * alias_preference

        phase_offsets = self._uniform(
            reference,
            0.0,
            2.0 * math.pi,
            (batch, mode_count, 1, 1),
            generator,
        )
        phases = 2.0 * math.pi * beat_coordinates + phase_offsets
        layout_sign = self._signed_uniform(
            reference, 1.0, 1.0, parameter_shape, generator
        )
        colour_phase_scale = self._uniform(
            reference,
            *self.COLOR_PHASE_SCALE_RANGE,
            parameter_shape,
            generator,
        )
        channel_offsets = (
            layout_order_phases
            * layout_sign.view(batch, 1, 1)
            * colour_phase_scale.view(batch, 1, 1)
        )
        anchor_beat = torch.gather(
            beat_coordinates.flatten(2),
            2,
            anchor_flat,
        ).squeeze(-1)
        relative_beat = beat_coordinates - anchor_beat.view(
            batch,
            mode_count,
            1,
            1,
        )
        chromatic_frequency_offsets = (
            self._sample_chromatic_frequency_offsets(
                reference,
                is_extreme=is_extreme,
                generator=generator,
            ).unsqueeze(1)
        )
        chromatic_dispersion_phase = (
            2.0
            * math.pi
            * relative_beat.unsqueeze(2)
            * chromatic_frequency_offsets
        )
        neutral = torch.cos(phases).unsqueeze(2)
        colour = torch.cos(
            phases.unsqueeze(2)
            + channel_offsets.unsqueeze(-1).unsqueeze(-1)
            + chromatic_dispersion_phase
        )
        achromatic_mix = self._uniform(
            reference,
            *profile["lattice_achromatic_mix"],
            parameter_shape,
            generator,
        ).unsqueeze(1)
        candidate_patterns = (
            (1.0 - achromatic_mix) * colour + achromatic_mix * neutral
        )

        mode_scores = visibility.mean(dim=(-2, -1)).clamp_min(1e-9)
        selection_random = self._uniform(
            reference,
            1e-5,
            1.0 - 1e-5,
            (batch, mode_count),
            generator,
        )
        gumbel = -torch.log(-torch.log(selection_random))
        selection_logits = mode_scores.log() + (
            self.LATTICE_MODE_RANDOMNESS * gumbel
        )
        primary_index = selection_logits.argmax(dim=1)
        primary_mask = F.one_hot(
            primary_index, num_classes=mode_count
        ).to(dtype=reference.dtype)
        secondary_logits = selection_logits.masked_fill(
            primary_mask.bool(),
            -torch.inf,
        )
        secondary_index = secondary_logits.argmax(dim=1)
        secondary_mask = F.one_hot(
            secondary_index, num_classes=mode_count
        ).to(dtype=reference.dtype)

        primary_mode_mask = primary_mask.view(batch, mode_count, 1, 1)
        secondary_mode_mask = secondary_mask.view(batch, mode_count, 1, 1)
        primary_pattern = (
            candidate_patterns * primary_mode_mask.unsqueeze(2)
        ).sum(dim=1)
        secondary_pattern = (
            candidate_patterns * secondary_mode_mask.unsqueeze(2)
        ).sum(dim=1)
        primary_visibility = (
            visibility * primary_mode_mask
        ).sum(dim=1, keepdim=True)
        secondary_visibility = (
            visibility * secondary_mode_mask
        ).sum(dim=1, keepdim=True)

        if self.lattice_visibility_floor is None:
            lattice_visibility_floor = self._uniform(
                reference,
                *profile["lattice_visibility_floor"],
                parameter_shape,
                generator,
            )
        else:
            lattice_visibility_floor = reference.new_full(
                parameter_shape,
                self.lattice_visibility_floor,
            )

        def normalized_visibility(value: torch.Tensor) -> torch.Tensor:
            maximum = value.amax(dim=(-2, -1), keepdim=True)
            normalized = (
                value / maximum.clamp_min(1e-8)
            ).clamp(0.0, 1.0).sqrt()
            relative_envelope = lattice_visibility_floor + (
                1.0 - lattice_visibility_floor
            ) * normalized
            if not self.preserve_absolute_visibility:
                return relative_envelope
            # Use the MTF as an amplitude response, not its square root.  A
            # fixed global calibration represents display modulation contrast
            # and camera/ISP contrast gain without renormalizing each image.
            absolute_gain = (
                self.ABSOLUTE_MTF_CONTRAST_GAIN * maximum
            ).clamp(0.0, 1.0)
            return absolute_gain * relative_envelope

        primary_envelope = normalized_visibility(primary_visibility)
        secondary_envelope = normalized_visibility(secondary_visibility)
        secondary_probability = self._secondary_order_probability(is_extreme)
        secondary_active = self._uniform(
            reference, 0.0, 1.0, parameter_shape, generator
        ).lt(secondary_probability).to(dtype=reference.dtype)
        secondary_ratio = secondary_active * self._uniform(
            reference,
            *profile["lattice_secondary_ratio"],
            parameter_shape,
            generator,
        )
        relative_visibility = torch.sigmoid(
            (
                torch.log(secondary_visibility.clamp_min(1e-8))
                - torch.log(primary_visibility.clamp_min(1e-8))
            )
            / self.LATTICE_SPATIAL_GATE_TEMPERATURE
        )
        secondary_weight = secondary_ratio * relative_visibility
        pattern = (
            (1.0 - secondary_weight)
            * primary_pattern
            * primary_envelope
            + secondary_weight
            * secondary_pattern
            * secondary_envelope
        )

        demosaiced = self._bayer_demosaic_pattern(
            pattern,
            generator,
            cfa_codes=cfa_codes,
        )
        cfa_mix_range = profile["lattice_cfa_mix"]
        cfa_mix = self._uniform(
            reference,
            min(1.0, cfa_mix_range[0] * self.cfa_mix_scale),
            min(1.0, cfa_mix_range[1] * self.cfa_mix_scale),
            parameter_shape,
            generator,
        )
        pattern = pattern + cfa_mix * (demosaiced - pattern)
        # Increase colour separation without changing the per-pixel luminance
        # component. This strengthens realistic RGB/CFA fringes while keeping
        # exposure loss and Moire contrast independently controllable.
        luminance_weights = reference.new_tensor(
            (0.2126, 0.7152, 0.0722)
        ).view(1, 3, 1, 1)
        luminance = (pattern * luminance_weights).sum(dim=1, keepdim=True)
        chroma_boost = self._uniform(
            reference,
            *profile["lattice_chroma_boost"],
            parameter_shape,
            generator,
        ) * self.chroma_scale * layout_chroma_gains.view(batch, 1, 1, 1)
        pattern = luminance + chroma_boost * (pattern - luminance)
        # Keep the selected sinusoid smooth. The former tanh normalization
        # generated extra harmonics that looked like additional screen rates.
        pattern = pattern - pattern.mean(dim=(-2, -1), keepdim=True)
        peak = pattern.abs().amax(
            dim=(1, 2, 3), keepdim=True
        )
        if self.preserve_absolute_visibility:
            # Clamp only over-range constructive/chroma peaks. Do not amplify
            # a weak MTF response back to unit contrast.
            normalizer = peak.clamp_min(1.0)
        else:
            normalizer = peak.clamp_min(1e-6)
        return pattern / normalizer

    def generate_moire(
        self,
        reference: torch.Tensor,
        is_extreme: bool = False,
        generator: Optional[torch.Generator] = None,
        projected_coordinates: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        optical_sigma: Optional[torch.Tensor] = None,
        display_layout_codes: Optional[torch.Tensor] = None,
        cfa_codes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate the metadata lattice, plus an optional legacy ablation."""

        batch, _, height, width = reference.shape
        if self.legacy_low_frequency_weight == 0.0:
            high_probability = self._high_frequency_probability(is_extreme)
            if high_probability == 0.0:
                return torch.zeros_like(reference)
            high_active = self._uniform(
                reference,
                0.0,
                1.0,
                (batch, 1, 1, 1),
                generator,
            ).lt(high_probability).to(dtype=reference.dtype)
            pattern = self.generate_high_frequency_moire(
                reference,
                is_extreme=is_extreme,
                generator=generator,
                projected_coordinates=projected_coordinates,
                optical_sigma=optical_sigma,
                display_layout_codes=display_layout_codes,
                cfa_codes=cfa_codes,
            )
            return high_active * pattern

        profile = self._profile_ranges(is_extreme)
        parameter_shape = (batch, 1, 1, 1)
        interference_probability = self._interference_probability(is_extreme)
        interference_active = self._uniform(
            reference,
            0.0,
            1.0,
            parameter_shape,
            generator,
        ).lt(interference_probability).to(dtype=reference.dtype)
        low_interaction_weight = interference_active * self._uniform(
            reference,
            *profile["low_wave_interaction"],
            parameter_shape,
            generator,
        ) * self.interference_scale
        high_interaction_weight = interference_active * self._uniform(
            reference,
            *profile["high_wave_interaction"],
            parameter_shape,
            generator,
        ) * self.interference_scale
        short_edge = float(min(height, width))
        half_short_edge = max(short_edge / 2.0, 1.0)
        phase_height = max(2, int(math.ceil(height * self.phase_scale)))
        phase_width = max(2, int(math.ceil(width * self.phase_scale)))
        phase_short_edge = float(min(phase_height, phase_width))
        beat_cycles_range = self._effective_beat_cycles_range(
            profile["beat_cycles"],
            phase_short_edge,
        )
        x, y, normalized_x, normalized_y = self._pixel_grid(
            reference,
            height,
            width,
            phase_height,
            phase_width,
        )

        # One shared display-to-camera geometry for all three colour channels.
        theta = self._uniform(
            reference,
            0.0,
            2.0 * math.pi,
            parameter_shape,
            generator,
        )
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        perpendicular = (-sin_theta * x + cos_theta * y) / half_short_edge

        optical_center_x = self._uniform(
            reference,
            -0.30,
            0.30,
            parameter_shape,
            generator,
        )
        optical_center_y = self._uniform(
            reference,
            -0.30,
            0.30,
            parameter_shape,
            generator,
        )
        optical_dx = normalized_x - optical_center_x
        optical_dy = normalized_y - optical_center_y
        radius_squared = 0.5 * (optical_dx.square() + optical_dy.square())

        radial_pixels = self._signed_uniform(
            reference,
            *profile["radial_pixels"],
            parameter_shape,
            generator,
        )
        radial_scale = radial_pixels / half_short_edge
        perspective_x = self._uniform(
            reference,
            *profile["perspective"],
            parameter_shape,
            generator,
        )
        perspective_y = self._uniform(
            reference,
            *profile["perspective"],
            parameter_shape,
            generator,
        )
        local_scale = (
            perspective_x * normalized_x + perspective_y * normalized_y
        )

        screen_x = x + x * local_scale + x * radial_scale * radius_squared
        screen_y = y + y * local_scale + y * radial_scale * radius_squared

        bend_pixels = self._signed_uniform(
            reference,
            *profile["bend_pixels"],
            parameter_shape,
            generator,
        )
        bend = bend_pixels * perpendicular.square()
        screen_x = screen_x + bend * cos_theta
        screen_y = screen_y + bend * sin_theta

        ripple_cycles = self._uniform(
            reference,
            0.35,
            1.25,
            parameter_shape,
            generator,
        )
        ripple_theta = self._uniform(
            reference,
            0.0,
            2.0 * math.pi,
            parameter_shape,
            generator,
        )
        ripple_phase = self._uniform(
            reference,
            0.0,
            2.0 * math.pi,
            parameter_shape,
            generator,
        )
        ripple_argument = 2.0 * math.pi * ripple_cycles * 0.5 * (
            torch.cos(ripple_theta) * normalized_x
            + torch.sin(ripple_theta) * normalized_y
        ) + ripple_phase
        ripple_pixels = self._signed_uniform(
            reference,
            *profile["ripple_pixels"],
            parameter_shape,
            generator,
        )
        ripple = ripple_pixels * torch.sin(ripple_argument)
        ripple_direction = self._uniform(
            reference,
            0.0,
            2.0 * math.pi,
            parameter_shape,
            generator,
        )
        screen_x = screen_x + ripple * torch.cos(ripple_direction)
        screen_y = screen_y + ripple * torch.sin(ripple_direction)

        primary_phase = self._difference_phase(
            reference,
            x,
            y,
            screen_x,
            screen_y,
            theta,
            short_edge,
            beat_cycles_range,
            generator,
        )
        secondary_theta = theta + (math.pi / 2.0) + self._uniform(
            reference,
            -math.pi / 12.0,
            math.pi / 12.0,
            parameter_shape,
            generator,
        )
        secondary_phase = self._difference_phase(
            reference,
            x,
            y,
            screen_x,
            screen_y,
            secondary_theta,
            short_edge,
            beat_cycles_range,
            generator,
        )

        # RGB stripes share the phases above but observe shifted display
        # subpixels. A randomized reversal covers RGB and BGR stripe layouts.
        base_offsets = reference.new_tensor(
            [-2.0 * math.pi / 3.0, 0.0, 2.0 * math.pi / 3.0]
        ).view(1, 3, 1, 1)
        layout_sign = self._signed_uniform(
            reference,
            1.0,
            1.0,
            parameter_shape,
            generator,
        )
        colour_phase_scale = self._uniform(
            reference,
            *self.COLOR_PHASE_SCALE_RANGE,
            parameter_shape,
            generator,
        )
        channel_offsets = base_offsets * layout_sign * colour_phase_scale
        achromatic_mix = self._uniform(
            reference,
            *self.ACHROMATIC_MIX_RANGE,
            parameter_shape,
            generator,
        )

        primary_colour = torch.cos(primary_phase + channel_offsets)
        primary_neutral = torch.cos(primary_phase)
        primary = (
            (1.0 - achromatic_mix) * primary_colour
            + achromatic_mix * primary_neutral
        )

        secondary_colour = torch.cos(secondary_phase - 0.65 * channel_offsets)
        secondary_neutral = torch.cos(secondary_phase)
        secondary = (
            (1.0 - achromatic_mix) * secondary_colour
            + achromatic_mix * secondary_neutral
        )
        secondary_weight = self._uniform(
            reference,
            *self.SECONDARY_WEIGHT_RANGE,
            parameter_shape,
            generator,
        )

        visibility_theta = self._uniform(
            reference,
            0.0,
            2.0 * math.pi,
            parameter_shape,
            generator,
        )
        visibility_phase = self._uniform(
            reference,
            0.0,
            2.0 * math.pi,
            parameter_shape,
            generator,
        )
        visibility_cycles = self._uniform(
            reference,
            0.45,
            1.15,
            parameter_shape,
            generator,
        )
        visibility_floor = self._uniform(
            reference,
            *profile["visibility_floor"],
            parameter_shape,
            generator,
        )
        visibility_wave = 0.5 + 0.5 * torch.cos(
            math.pi
            * visibility_cycles
            * (
                torch.cos(visibility_theta) * normalized_x
                + torch.sin(visibility_theta) * normalized_y
            )
            + visibility_phase
        )
        # The slow envelope creates broad strong/weak zones. Real captures
        # rarely exhibit a spatially uniform Moire magnitude.
        visibility = visibility_floor + (
            1.0 - visibility_floor
        ) * visibility_wave

        linear_pattern = (
            (primary + secondary_weight * secondary)
            / (1.0 + secondary_weight)
        )
        # Multiplication produces the phase-sum and phase-difference terms of
        # the two wave families. This creates genuine constructive/destructive
        # intersections instead of a transparent overlay of two stripe maps.
        low_interaction = primary * secondary
        low_interaction = low_interaction - low_interaction.mean(
            dim=(-2, -1),
            keepdim=True,
        )
        interaction_peak = low_interaction.abs().amax(
            dim=(1, 2, 3),
            keepdim=True,
        ).clamp_min(1e-6)
        low_interaction = low_interaction / interaction_peak
        pattern = linear_pattern + low_interaction_weight * low_interaction
        pattern = pattern - pattern.mean(dim=(-2, -1), keepdim=True)
        peak = pattern.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        pattern = pattern / peak
        if (phase_height, phase_width) != (height, width):
            pattern = F.interpolate(
                pattern,
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            )
            pattern = pattern - pattern.mean(dim=(-2, -1), keepdim=True)
            peak = pattern.abs().amax(
                dim=(1, 2, 3),
                keepdim=True,
            ).clamp_min(1e-6)
            pattern = pattern / peak

        high_probability = self._high_frequency_probability(is_extreme)
        if high_probability > 0.0:
            high_active = self._uniform(
                reference,
                0.0,
                1.0,
                parameter_shape,
                generator,
            ).lt(high_probability).to(dtype=reference.dtype)
            high_weight = high_active * self._uniform(
                reference,
                *profile["high_mix"],
                parameter_shape,
                generator,
            )
            high_pattern = self.generate_high_frequency_moire(
                reference,
                is_extreme=is_extreme,
                generator=generator,
                projected_coordinates=projected_coordinates,
                optical_sigma=optical_sigma,
                display_layout_codes=display_layout_codes,
                cfa_codes=cfa_codes,
            )
            # The fine alias is an additional camera/display beat family.  It
            # must not replace the broad two-dimensional pattern: convex
            # interpolation made large high-frequency draws erase the very
            # screen-spanning interference that they should enrich.
            linear_mix = (
                self.legacy_low_frequency_weight * pattern
                + high_weight * high_pattern
            )
            # The product convolves both spectra and therefore introduces
            # visible high/low sum-and-difference sidebands around crossings.
            high_interaction = pattern * high_pattern
            high_interaction = high_interaction - high_interaction.mean(
                dim=(-2, -1),
                keepdim=True,
            )
            interaction_peak = high_interaction.abs().amax(
                dim=(1, 2, 3),
                keepdim=True,
            ).clamp_min(1e-6)
            high_interaction = high_interaction / interaction_peak
            pattern = linear_mix + (
                high_active
                * high_interaction_weight
                * high_interaction
            )
            # A soft sensor/display response limits rare constructive peaks
            # without globally shrinking the broad carrier.  Peak-only
            # normalization allowed a few high-frequency crossings to reduce
            # the screen-wide beat field to less than half its original RMS.
            pattern = torch.tanh(pattern)
            pattern = pattern - pattern.mean(dim=(-2, -1), keepdim=True)
            peak = pattern.abs().amax(
                dim=(1, 2, 3),
                keepdim=True,
            ).clamp_min(1e-6)
            pattern = pattern / peak

        if (phase_height, phase_width) != (height, width):
            visibility = F.interpolate(
                visibility,
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            )
        channel_response = self._uniform(
            reference,
            *profile["channel_response"],
            (batch, 3, 1, 1),
            generator,
        )
        channel_response = channel_response / channel_response.mean(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)
        pattern = pattern * visibility * channel_response
        pattern = pattern - pattern.mean(dim=(-2, -1), keepdim=True)
        peak = pattern.abs().amax(
            dim=(1, 2, 3),
            keepdim=True,
        ).clamp_min(1e-6)
        pattern = pattern / peak
        return pattern

    def _difference_phase(
        self,
        reference: torch.Tensor,
        sensor_x: torch.Tensor,
        sensor_y: torch.Tensor,
        screen_x: torch.Tensor,
        screen_y: torch.Tensor,
        theta: torch.Tensor,
        short_edge: float,
        beat_cycles_range: Tuple[float, float],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Evaluate the low-pass display/sensor phase difference directly."""

        batch = reference.shape[0]
        parameter_shape = (batch, 1, 1, 1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        sensor_coordinate = cos_theta * sensor_x + sin_theta * sensor_y
        display_coordinate = cos_theta * screen_x + sin_theta * screen_y

        sensor_frequency = self._uniform(
            reference,
            *self.SENSOR_FREQUENCY_RANGE,
            parameter_shape,
            generator,
        )
        beat_cycles = self._uniform(
            reference,
            *beat_cycles_range,
            parameter_shape,
            generator,
        )
        beat_sign = self._signed_uniform(
            reference,
            1.0,
            1.0,
            parameter_shape,
            generator,
        )
        display_frequency = sensor_frequency + beat_sign * (
            beat_cycles / short_edge
        )
        phase_offset = self._uniform(
            reference,
            0.0,
            2.0 * math.pi,
            parameter_shape,
            generator,
        )
        return 2.0 * math.pi * (
            display_frequency * display_coordinate
            - sensor_frequency * sensor_coordinate
        ) + phase_offset

    def _content_adaptive_gain(self, image: torch.Tensor) -> torch.Tensor:
        """Return a smooth, phase-neutral visibility gain from image content.

        Moire is perceptually exposed over sky, walls, and other low-gradient
        regions, while object contours and fine texture mask part of the same
        physical modulation. The gain is deliberately bounded and contains no
        periodic term, so it cannot introduce an unrelated frequency family.
        """

        gain_shape = (image.shape[0], 1, image.shape[-2], image.shape[-1])
        if self.content_adaptive_strength == 0.0:
            return image.new_ones(gain_shape)

        luminance_weights = image.new_tensor(
            (0.2126, 0.7152, 0.0722)
        ).view(1, 3, 1, 1)
        luminance = (image * luminance_weights).sum(dim=1, keepdim=True)
        horizontal = F.pad(
            (luminance[..., 1:] - luminance[..., :-1]).abs(),
            (0, 1, 0, 0),
            mode="replicate",
        )
        vertical = F.pad(
            (luminance[..., 1:, :] - luminance[..., :-1, :]).abs(),
            (0, 0, 0, 1),
            mode="replicate",
        )
        gradient = (horizontal.square() + vertical.square() + 1e-12).sqrt()
        local_gradient = F.avg_pool2d(
            gradient,
            kernel_size=5,
            stride=1,
            padding=2,
            count_include_pad=False,
        )
        smoothness = torch.exp(
            -self.CONTENT_EDGE_SENSITIVITY * local_gradient
        ).clamp(0.0, 1.0)
        return 1.0 + self.content_adaptive_strength * (
            smoothness - self.CONTENT_ADAPTIVE_NEUTRAL_SMOOTHNESS
        )

    def sample_spatial_envelope(
        self,
        reference: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Sample RGB-shared region coverage in [0,1], shaped [B,1,H,W].

        Each local sample contains 1--N rotated, irregular superellipse regions
        with independent centres, axis scales, shapes, and opacity. Compact
        support and smoothstep edges leave exact zeros outside the regions.
        This is an augmentation envelope, not an inferred optical field.
        """
        reference = self._working_image(reference)
        batch, _, height, width = reference.shape
        if self.spatial_coverage == "global":
            # Preserve the old full-frame RNG sequence and modulation exactly.
            return reference.new_ones((batch, 1, height, width))
        count_min, count_max = self.region_count_range
        shape = (batch, count_max, 1, 1)

        def uniform(low: float, high: float) -> torch.Tensor:
            return self._uniform(reference, low, high, shape, generator)

        centre_x = uniform(0, float(width - 1))
        centre_y = uniform(0, float(height - 1))
        radius_x = (uniform(*self.region_scale_range) * width).clamp_min(1.0)
        radius_y = (uniform(*self.region_scale_range) * height).clamp_min(1.0)
        angle = uniform(0, 2 * math.pi)
        exponent = uniform(1.6, 3.2)
        boundary_phase = uniform(0, 2 * math.pi)
        opacity = uniform(0.55, 1.0)
        x = torch.arange(width, device=reference.device, dtype=reference.dtype).view(1, 1, 1, width)
        y = torch.arange(height, device=reference.device, dtype=reference.dtype).view(1, 1, height, 1)
        dx, dy = x - centre_x, y - centre_y
        u = (angle.cos() * dx + angle.sin() * dy) / radius_x
        v = (-angle.sin() * dx + angle.cos() * dy) / radius_y
        radius = (u.abs().pow(exponent) + v.abs().pow(exponent)).pow(1.0 / exponent)
        theta = torch.atan2(v, u)
        boundary = 1.0 + 0.12 * torch.cos(3 * theta + boundary_phase) + 0.06 * torch.sin(5 * theta - boundary_phase)
        transition = ((radius / boundary - (1 - self.region_softness)) / self.region_softness).clamp(0, 1)
        regions = 1.0 - transition.square() * (3.0 - 2.0 * transition)
        counts = torch.randint(count_min, count_max + 1, (batch, 1, 1, 1), device=reference.device, generator=generator)
        indices = torch.arange(count_max, device=reference.device).view(1, count_max, 1, 1)
        regions = regions * opacity * (indices < counts).to(reference.dtype)
        # Smooth union: overlaps remain bounded and do not create hard seams.
        envelope = 1.0 - (1.0 - regions).prod(dim=1, keepdim=True)
        if self.spatial_coverage == "mixed":
            full = self._uniform(reference, 0, 1, (batch, 1, 1, 1), generator) < self.global_coverage_probability
            envelope = torch.where(full, torch.ones_like(envelope), envelope)
        return envelope.clamp(0, 1)

    def forward(
        self,
        img: torch.Tensor,
        is_extreme: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        is_extreme = self._resolve_is_extreme(is_extreme)
        self._validate_image(img)
        original_dtype = img.dtype
        working = self._working_image(img).clamp(0.0, 1.0)

        # Sample the camera/display geometry and device profiles once.  The
        # rectified content residual, reciprocal-lattice alias, optical PSF,
        # display sub-pixel layout, and sensor CFA must describe one capture,
        # rather than unrelated random transforms applied to separate layers.
        projected_coordinates = self.projected_lattice_coordinates(
            working,
            is_extreme=is_extreme,
            generator=generator,
        )
        warped = self._warp_content_with_projected_residual(
            working,
            projected_coordinates,
            is_extreme=is_extreme,
            generator=generator,
        )
        optical_sigma = self._sample_optical_sigma(
            warped,
            is_extreme=is_extreme,
            generator=generator,
        )
        display_layout_codes = self._sample_display_layout_codes(
            warped,
            generator,
        )
        cfa_codes = self._sample_cfa_codes(warped, generator)
        pattern = self.generate_moire(
            warped,
            is_extreme=is_extreme,
            generator=generator,
            projected_coordinates=projected_coordinates,
            optical_sigma=optical_sigma,
            display_layout_codes=display_layout_codes,
            cfa_codes=cfa_codes,
        )
        profile = self._profile_ranges(is_extreme)
        strength = self._uniform(
            warped,
            *(self.modulation_strength_range or profile["strength"]),
            (warped.shape[0], 1, 1, 1),
            generator,
        ) * self.strength_scale
        severe_probability = self._severe_contrast_probability(
            is_extreme
        )
        if severe_probability > 0.0:
            severe_active = self._uniform(
                warped,
                0.0,
                1.0,
                (warped.shape[0], 1, 1, 1),
                generator,
            ).lt(severe_probability).to(dtype=working.dtype)
            severe_multiplier = self._uniform(
                warped,
                *profile["severe_strength_multiplier"],
                (warped.shape[0], 1, 1, 1),
                generator,
            )
            strength = strength * (
                1.0
                + severe_active * (severe_multiplier - 1.0)
            )
        illumination = self.generate_illumination(
            warped,
            is_extreme=is_extreme,
            generator=generator,
        )
        content_gain = self._content_adaptive_gain(warped)
        spatial_envelope = self.sample_spatial_envelope(warped, generator)
        localized_pattern = pattern * spatial_envelope

        if self.linear_light:
            signal = self._srgb_to_linear(warped)
            moire_gain = 1.0 + strength * content_gain * localized_pattern
            if self.exposure_scale == 0.0:
                modulated_linear = signal * moire_gain
            else:
                modulated_linear = signal * illumination * moire_gain
            modulated = self._apply_linear_capture_pipeline(
                modulated_linear.clamp(0.0, 1.0),
                is_extreme=is_extreme,
                generator=generator,
                optical_sigma=optical_sigma,
                cfa_codes=cfa_codes,
            )
        else:
            # If sRGB is approximated as linear**(1/gamma), applying gain in
            # linear light becomes gain**(1/gamma) in sRGB. The first-order
            # expansion below avoids a full-image power while remaining close
            # over the configured contrast range.
            moire_gain = 1.0 + (strength / 2.2) * content_gain * localized_pattern
            if self.exposure_scale == 0.0:
                modulated_srgb = warped * moire_gain
            else:
                exposure_srgb = illumination.pow(1.0 / 2.2)
                modulated_srgb = warped * exposure_srgb * moire_gain
            modulated = self._apply_capture_lighting(
                modulated_srgb,
                is_extreme=is_extreme,
                generator=generator,
                optical_sigma=optical_sigma,
                cfa_codes=cfa_codes,
            )
        if (self.spatial_coverage != "global" and
                self.exposure_scale == self.capture_lighting_scale ==
                self.content_warp_scale == self.optical_psf_scale == self.sensor_isp_scale == 0):
            # The pure-corruption preset leaves unsupported pixels bit-exact,
            # rather than passing them through an sRGB round trip.
            modulated = torch.where(spatial_envelope > 0, modulated, working)
        result = modulated.clamp(0.0, 1.0).to(dtype=original_dtype)
        if return_details:
            return result, {
                "spatial_envelope": spatial_envelope,
                "modulation_strength": strength,
                "moire_pattern": pattern,
            }
        return result


class PhysicalMoire(nn.Module):
    """MBRS-compatible object wrapper around the physical simulator.

    MBRS noise objects receive ``[encoded_image, cover_image]`` and return the
    attacked encoded image.  The cover is intentionally ignored because this
    screen-camera simulator is blind and acts only on the displayed image.

    A plain image tensor is accepted as a convenience, which also makes this
    object usable as the sole item in a standard ``nn.Sequential``.  Random
    attack parameters are sampled on every call unless an explicit
    ``torch.Generator`` is supplied to ``forward``.

    Args:
        is_extreme: Select the simulator's normal or extreme attack profile.
        input_range: ``"minus_one_one"`` for MBRS tensors in ``[-1, 1]`` or
            ``"zero_one"`` for tensors in ``[0, 1]``. Inputs are clamped to
            the declared range before simulation. The output uses the same
            range as the input contract.
        **simulator_options: Forwarded to :class:`EfficientScreenMoireNoise`.
    """

    INPUT_RANGES = frozenset({"minus_one_one", "zero_one"})

    def __init__(
        self,
        is_extreme: bool = True,
        input_range: str = "minus_one_one",
        **simulator_options: Any,
    ) -> None:
        super().__init__()
        if not isinstance(is_extreme, bool):
            raise TypeError("is_extreme must be bool")
        input_range = str(input_range).lower()
        if input_range not in self.INPUT_RANGES:
            raise ValueError(
                "input_range must be 'minus_one_one' or 'zero_one'"
            )
        self.is_extreme = is_extreme
        self.input_range = input_range
        # The adapter clamps and maps the declared external range itself.
        # Avoid a redundant reduction and GPU synchronization in the wrapped
        # simulator unless the caller explicitly requests validation.
        simulator_options.setdefault("validate_input", False)
        simulator_options.setdefault("default_is_extreme", is_extreme)
        self.simulator = EfficientScreenMoireNoise(**simulator_options)

    @staticmethod
    def _encoded_image(image_and_cover: object) -> torch.Tensor:
        if torch.is_tensor(image_and_cover):
            return image_and_cover
        if not isinstance(image_and_cover, (list, tuple)):
            raise TypeError(
                "PhysicalMoire expects an image tensor or the MBRS pair "
                "[encoded_image, cover_image]"
            )
        if len(image_and_cover) != 2:
            raise ValueError(
                "the MBRS input must contain exactly encoded_image and "
                "cover_image"
            )
        encoded_image = image_and_cover[0]
        if not torch.is_tensor(encoded_image):
            raise TypeError("encoded_image must be a torch.Tensor")
        return encoded_image

    def forward(
        self,
        image_and_cover: object,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        encoded_image = self._encoded_image(image_and_cover)
        if self.input_range == "minus_one_one":
            simulator_input = encoded_image.clamp(-1.0, 1.0)
            simulator_input = (simulator_input + 1.0) * 0.5
        else:
            simulator_input = encoded_image.clamp(0.0, 1.0)
        attacked = self.simulator(
            simulator_input,
            is_extreme=self.is_extreme,
            generator=generator,
        )
        if self.input_range == "minus_one_one":
            attacked = attacked * 2.0 - 1.0
        return attacked

    def extra_repr(self) -> str:
        profile = "extreme" if self.is_extreme else "normal"
        return f"profile={profile}, input_range={self.input_range}"


def _resolve_standalone_device(requested: object) -> torch.device:
    """Resolve a CLI/programmatic device without repository utilities."""

    if str(requested).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch installation has no "
            "available CUDA device"
        )
    return device


def simulate_screen_capture_file(
    input_path: object,
    output_path: object,
    *,
    device: object = "auto",
    is_extreme: bool = True,
    seed: Optional[int] = None,
    overwrite: bool = False,
    preset: str = "screen_capture",
    severity: str = "random",
    **simulator_options: Any,
) -> Dict[str, Any]:
    """Read one RGB image, synthesize a screen capture, and save the result.

    Pillow is imported only for this image-file convenience API.  Importing
    and using :class:`EfficientScreenMoireNoise` on tensors still requires only
    PyTorch and the Python standard library.
    """

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required for image-file input/output; install it with "
            "`python -m pip install Pillow`"
        ) from error

    source = Path(input_path).expanduser().resolve(strict=True)
    destination = Path(output_path).expanduser().resolve(strict=False)
    if source == destination:
        raise ValueError("output_path must differ from input_path")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {destination}; pass overwrite=True to replace it"
        )

    resolved_device = _resolve_standalone_device(device)
    with Image.open(source) as opened:
        rgb = opened.convert("RGB")
        width, height = rgb.size
        pixel_buffer = bytearray(rgb.tobytes())

    image = torch.frombuffer(pixel_buffer, dtype=torch.uint8).reshape(
        height,
        width,
        3,
    )
    image = (
        image.permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=resolved_device, dtype=torch.float32)
        .div_(255.0)
    )
    if preset == "image_corruption":
        simulator = EfficientScreenMoireNoise.for_image_corruption(
            severity=severity, device=resolved_device, **simulator_options
        ).eval()
    elif preset == "screen_capture":
        simulator = EfficientScreenMoireNoise(
            device=resolved_device, **simulator_options
        ).eval()
    else:
        raise ValueError("preset must be 'screen_capture' or 'image_corruption'")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=resolved_device).manual_seed(
            int(seed)
        )

    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    import time

    started = time.perf_counter()
    with torch.inference_mode():
        distorted = simulator(
            image,
            is_extreme=bool(is_extreme),
            generator=generator,
        )
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    pixels = (
        distorted[0]
        .mul(255.0)
        .round()
        .clamp(0.0, 255.0)
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .contiguous()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_image = Image.frombytes(
        "RGB",
        (width, height),
        pixels.numpy().tobytes(),
    )
    output_image.save(destination)
    return {
        "input_path": str(source),
        "output_path": str(destination),
        "width": width,
        "height": height,
        "device": str(resolved_device),
        "mode": "extreme" if is_extreme else "normal",
        "seed": None if seed is None else int(seed),
        "synthesis_ms": elapsed_ms,
        "capture_metadata": simulator.capture_metadata_path,
        "preset": preset,
        "severity": severity if preset == "image_corruption" else None,
        "strength_scale": simulator.strength_scale,
        "alias_period_scale": simulator.alias_period_scale,
        "chroma_scale": simulator.chroma_scale,
        "spatial_coverage": simulator.spatial_coverage,
        "region_count_range": simulator.region_count_range,
        "region_scale_range": simulator.region_scale_range,
        "modulation_strength_range": simulator.modulation_strength_range,
        "global_coverage_probability": simulator.global_coverage_probability,
    }


def _standalone_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate one physically inspired simulated screen-camera image "
            "using this standalone file."
        )
    )
    parser.add_argument("input", type=Path, help="input image path")
    parser.add_argument("output", type=Path, help="output image path")
    parser.add_argument(
        "--preset",
        choices=("screen_capture", "image_corruption"),
        default="screen_capture",
        help="screen-camera augmentation or shape-preserving Moire corruption",
    )
    parser.add_argument(
        "--severity", choices=("random", "light", "medium", "strong"), default="random",
        help="modulation strength for the image_corruption preset",
    )
    parser.add_argument(
        "--mode",
        choices=("normal", "extreme"),
        default="extreme",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete PyTorch device such as cuda:0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "optional reproducibility seed; when omitted, attack parameters "
            "are sampled randomly"
        ),
    )
    parser.add_argument("--strength-scale", type=float, default=None)
    parser.add_argument("--exposure-scale", type=float, default=None)
    parser.add_argument("--period-scale", type=float, default=1.0)
    parser.add_argument("--chroma-scale", type=float, default=1.0)
    parser.add_argument("--coverage", choices=("global", "local", "mixed"))
    parser.add_argument("--strength-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--region-count", type=int, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--region-scale", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--global-probability", type=float)
    parser.add_argument(
        "--capture-metadata",
        type=Path,
        help="optional custom capture JSON; the embedded profile is the default",
    )
    parser.add_argument(
        "--exact-linear-light",
        action="store_true",
        help="use the slower exact linear-light capture path",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )
    args = parser.parse_args()
    options = {
        "capture_metadata_path": args.capture_metadata,
        "alias_period_scale": args.period_scale,
        "chroma_scale": args.chroma_scale,
    }
    # Unspecified flags must not overwrite the selected preset's defaults.
    if args.strength_scale is not None:
        options["strength_scale"] = args.strength_scale
    if args.exposure_scale is not None:
        options["exposure_scale"] = args.exposure_scale
    if args.exact_linear_light:
        options["linear_light"] = True
    for option, value in (
        ("spatial_coverage", args.coverage),
        ("modulation_strength_range", args.strength_range),
        ("region_count_range", args.region_count),
        ("region_scale_range", args.region_scale),
        ("global_coverage_probability", args.global_probability),
    ):
        if value is not None:
            options[option] = value
    result = simulate_screen_capture_file(
        args.input,
        args.output,
        device=args.device,
        is_extreme=args.mode == "extreme",
        seed=args.seed,
        overwrite=args.overwrite,
        preset=args.preset,
        severity=args.severity,
        **options,
    )
    print(
        "Generated {width}x{height} {preset} ({mode}) on {device} in "
        "{synthesis_ms:.3f} ms: {output_path}".format(**result)
    )


__all__ = [
    "EfficientScreenMoireNoise",
    "PhysicalMoire",
    "simulate_screen_capture_file",
]


if __name__ == "__main__":
    _standalone_main()
