"""Standalone WeChat-like worst-case JPEG compressor.

Copy this file into any Python project and install Pillow:

    pip install pillow

It does not depend on any local modules. The compressor is designed for
stress-testing image pipelines with a deterministic approximation of the
stronger common WeChat compression path observed in the collected samples.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFilter, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


try:
    _RESAMPLING_ENUM = Image.Resampling
except AttributeError:  # Pillow < 9.1 compatibility
    _RESAMPLING_ENUM = Image


RESAMPLING = {
    "nearest": _RESAMPLING_ENUM.NEAREST,
    "box": _RESAMPLING_ENUM.BOX,
    "bilinear": _RESAMPLING_ENUM.BILINEAR,
    "hamming": _RESAMPLING_ENUM.HAMMING,
    "bicubic": _RESAMPLING_ENUM.BICUBIC,
    "lanczos": _RESAMPLING_ENUM.LANCZOS,
}


STD_LUMA = [
    16,
    11,
    10,
    16,
    24,
    40,
    51,
    61,
    12,
    12,
    14,
    19,
    26,
    58,
    60,
    55,
    14,
    13,
    16,
    24,
    40,
    57,
    69,
    56,
    14,
    17,
    22,
    29,
    51,
    87,
    80,
    62,
    18,
    22,
    37,
    56,
    68,
    109,
    103,
    77,
    24,
    35,
    55,
    64,
    81,
    104,
    113,
    92,
    49,
    64,
    78,
    87,
    103,
    121,
    120,
    101,
    72,
    92,
    95,
    98,
    112,
    100,
    103,
    99,
]


STD_CHROMA = [
    17,
    18,
    24,
    47,
    99,
    99,
    99,
    99,
    18,
    21,
    26,
    66,
    99,
    99,
    99,
    99,
    24,
    26,
    56,
    99,
    99,
    99,
    99,
    99,
    47,
    66,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
    99,
]


def scaled_table(std_table: list[int], quality: int) -> list[int]:
    """Scale a standard IJG/libjpeg quantization table for a JPEG quality value."""

    if not 1 <= quality <= 100:
        raise ValueError(f"Quality must be in [1, 100], got {quality}")
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    output = []
    for value in std_table:
        scaled = int((value * scale + 50) // 100)
        output.append(max(1, min(255, scaled)))
    return output


OBSERVED_WECHAT_QUALITY_PRESETS = {
    # Strongest common bucket in the collected WeChat corpus.
    # q60/q85/q70/q76 cover about 94.8% of unique images; q60 is the most destructive among them.
    "mainstream_worst": 60,
    # Rare but still observed in the collected corpus. Useful for conservative stress testing.
    "aggressive": 46,
    # Extreme outlier in the collected corpus. Use only when you want an upper-bound damage test.
    "ultra": 10,
    # Common non-worst reference presets.
    "q85": 85,
    "q76": 76,
    "q70": 70,
    "q60": 60,
    "q46": 46,
    "q10": 10,
}


__all__ = [
    "IMAGE_SUFFIXES",
    "OBSERVED_WECHAT_QUALITY_PRESETS",
    "RESAMPLING",
    "STD_CHROMA",
    "STD_LUMA",
    "WeChatWorstCaseCompressor",
    "WeChatWorstCaseConfig",
    "choose_wechat_like_size",
    "compress_image",
    "is_close_aspect",
    "scaled_table",
]


@dataclass(frozen=True)
class WeChatWorstCaseConfig:
    quality: int = 60
    short_edge: int = 1280
    four_three_landscape_size: tuple[int, int] = (1706, 1279)
    four_three_portrait_size: tuple[int, int] = (1279, 1706)
    four_three_tolerance: float = 0.015
    upscale_small_images: bool = False
    resample: str = "hamming"
    blur_radius: float = 0.6
    subsampling: int = 2
    optimize_huffman: bool = False


def is_close_aspect(width: int, height: int, target_aspect: float, tolerance: float) -> bool:
    return abs(width / height - target_aspect) <= tolerance


def choose_wechat_like_size(width: int, height: int, config: WeChatWorstCaseConfig) -> tuple[int, int]:
    """Approximate WeChat's common large-photo resize rule.

    The collected corpus is dominated by 4:3 photos resized to 1706x1279 or 1707x1280.
    For non-4:3 images, WeChat-like outputs often keep the short side around 1280
    while preserving aspect ratio, e.g. panoramic images around 2779x1280.
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {width}x{height}")

    if is_close_aspect(width, height, 4 / 3, config.four_three_tolerance):
        return config.four_three_landscape_size
    if is_close_aspect(width, height, 3 / 4, config.four_three_tolerance):
        return config.four_three_portrait_size

    short_edge = min(width, height)
    if short_edge <= config.short_edge and not config.upscale_small_images:
        return width, height

    scale = config.short_edge / short_edge
    return max(1, int(width * scale)), max(1, int(height * scale))


class WeChatWorstCaseCompressor:
    """Worst-case WeChat-like JPEG compressor.

    This compressor is intentionally deterministic. It uses the most destructive
    common WeChat quantization bucket by default, plus the same structural choices
    observed in the collected corpus: 4:2:0 subsampling, IJG/libjpeg-style standard
    quantization tables, metadata-stripped baseline JPEG, and a resize/low-pass
    approximation calibrated on paired WeChat samples.
    """

    def __init__(self, config: WeChatWorstCaseConfig | None = None):
        self.config = config or WeChatWorstCaseConfig()
        if self.config.resample not in RESAMPLING:
            raise ValueError(f"Unsupported resample mode: {self.config.resample}")
        if not 1 <= self.config.quality <= 100:
            raise ValueError(f"Quality must be in [1, 100], got {self.config.quality}")

    @classmethod
    def from_preset(cls, preset: str = "mainstream_worst", **overrides: object) -> "WeChatWorstCaseCompressor":
        if preset not in OBSERVED_WECHAT_QUALITY_PRESETS:
            choices = ", ".join(sorted(OBSERVED_WECHAT_QUALITY_PRESETS))
            raise ValueError(f"Unknown preset {preset!r}. Available presets: {choices}")
        config_data = asdict(WeChatWorstCaseConfig())
        config_data["quality"] = OBSERVED_WECHAT_QUALITY_PRESETS[preset]
        config_data.update(overrides)
        return cls(WeChatWorstCaseConfig(**config_data))

    @property
    def qtables(self) -> dict[int, list[int]]:
        return {
            0: scaled_table(STD_LUMA, self.config.quality),
            1: scaled_table(STD_CHROMA, self.config.quality),
        }

    def target_size_for_image(self, image: Image.Image) -> tuple[int, int]:
        return choose_wechat_like_size(image.width, image.height, self.config)

    def preprocess(self, source: str | Path | Image.Image) -> Image.Image:
        if isinstance(source, Image.Image):
            image = source
        else:
            image = Image.open(source)
        image = ImageOps.exif_transpose(image).convert("RGB")
        target_size = self.target_size_for_image(image)
        if image.size != target_size:
            image = image.resize(target_size, RESAMPLING[self.config.resample])
        if self.config.blur_radius > 0:
            image = image.filter(ImageFilter.GaussianBlur(radius=self.config.blur_radius))
        return image

    def encode_to_bytes(self, source: str | Path | Image.Image) -> bytes:
        image = self.preprocess(source)
        buffer = BytesIO()
        image.save(
            buffer,
            "JPEG",
            qtables=self.qtables,
            subsampling=self.config.subsampling,
            optimize=self.config.optimize_huffman,
        )
        return buffer.getvalue()

    def compress(self, source: str | Path | Image.Image, output: str | Path) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(self.encode_to_bytes(source))
        return output

    def compress_to_image(self, source: str | Path | Image.Image) -> Image.Image:
        return Image.open(BytesIO(self.encode_to_bytes(source))).convert("RGB")

    def describe(self, source: str | Path | Image.Image | None = None) -> dict[str, object]:
        data: dict[str, object] = {
            "config": asdict(self.config),
            "luma_quantization_table": self.qtables[0],
            "chroma_quantization_table": self.qtables[1],
            "subsampling": "4:2:0" if self.config.subsampling == 2 else self.config.subsampling,
            "interpretation": "mainstream worst-case WeChat-like compression"
            if self.config.quality == 60
            else "aggressive WeChat-like stress-test compression",
        }
        if source is not None:
            if isinstance(source, Image.Image):
                image = source
            else:
                image = Image.open(source)
            image = ImageOps.exif_transpose(image)
            data["source_size"] = [image.width, image.height]
            data["target_size"] = list(self.target_size_for_image(image))
        return data


def compress_image(
    source: str | Path | Image.Image,
    output: str | Path,
    preset: str = "mainstream_worst",
    **overrides: object,
) -> Path:
    """Compress one image with a WeChat-like preset and write a JPEG file.

    Example:
        compress_image("input.png", "output.jpg", preset="mainstream_worst")
    """

    compressor = WeChatWorstCaseCompressor.from_preset(preset, **overrides)
    return compressor.compress(source, output)


def iter_input_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix: {path}")
        yield path
        return
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            yield candidate


def output_path_for(input_path: Path, input_root: Path, output: Path) -> Path:
    if input_root.is_file():
        return output
    relative = input_path.relative_to(input_root)
    return output / relative.with_suffix(".jpg")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply worst-case WeChat-like JPEG compression.")
    parser.add_argument("input", type=Path, help="Input image file or directory.")
    parser.add_argument("output", type=Path, help="Output JPEG path, or output directory for batch input.")
    parser.add_argument(
        "--preset",
        choices=sorted(OBSERVED_WECHAT_QUALITY_PRESETS),
        default="mainstream_worst",
        help="mainstream_worst=q60, aggressive=q46, ultra=q10.",
    )
    parser.add_argument("--quality", type=int, default=None, help="Override preset quality.")
    parser.add_argument("--short-edge", type=int, default=1280)
    parser.add_argument("--blur-radius", type=float, default=0.6)
    parser.add_argument("--resample", choices=sorted(RESAMPLING), default="hamming")
    parser.add_argument("--upscale-small-images", action="store_true")
    parser.add_argument("--optimize-huffman", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    overrides: dict[str, object] = {
        "short_edge": args.short_edge,
        "blur_radius": args.blur_radius,
        "resample": args.resample,
        "upscale_small_images": args.upscale_small_images,
        "optimize_huffman": args.optimize_huffman,
    }
    if args.quality is not None:
        overrides["quality"] = args.quality
    compressor = WeChatWorstCaseCompressor.from_preset(args.preset, **overrides)

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    rows = []
    for source in iter_input_images(input_path):
        target = output_path_for(source, input_path, output_path)
        compressor.compress(source, target)
        info = compressor.describe(source)
        rows.append(
            {
                "source": str(source),
                "output": str(target),
                "source_size": info["source_size"],
                "target_size": info["target_size"],
                "quality": compressor.config.quality,
                "output_bytes": target.stat().st_size,
            }
        )

    if args.write_report:
        report_path = output_path.with_suffix(".json") if input_path.is_file() else output_path / "wechat_worst_case_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "compressor": compressor.describe(),
                    "count": len(rows),
                    "outputs": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps({"count": len(rows), "preset": args.preset, "quality": compressor.config.quality}, ensure_ascii=False))


if __name__ == "__main__":
    main()
