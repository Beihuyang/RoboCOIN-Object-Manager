"""Resident 2K Real-ESRGAN preprocessing for SAM3 image inference."""

from __future__ import annotations

import gc
import hashlib
import io
import os
import sys
import threading
import types
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "realesrgan" / "RealESRGAN_x2plus.pth"
MODEL_SHA256 = "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
MODEL_SCALE = 2
TARGET_LONG_EDGE = 2048
TILE_CANDIDATES = (1024, 768, 512, 384, 256, 192, 128, 96, 64)


def _enabled() -> bool:
    return os.environ.get("ROBOCOIN_SUPER_RESOLUTION", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _install_torchvision_compatibility() -> None:
    """Adapt BasicSR 1.4.2 to TorchVision versions that moved this module."""
    name = "torchvision.transforms.functional_tensor"
    if name in sys.modules:
        return
    from torchvision.transforms.functional import rgb_to_grayscale

    compatibility = types.ModuleType(name)
    compatibility.rgb_to_grayscale = rgb_to_grayscale
    sys.modules[name] = compatibility


def _verify_model() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Real-ESRGAN weights not found: {MODEL_PATH}. Run ./setup.sh first."
        )
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(
            f"Real-ESRGAN weight checksum mismatch: {MODEL_PATH}"
        )


def check_dependencies() -> None:
    """Verify inference imports without loading weights onto the GPU."""
    _install_torchvision_compatibility()
    from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401
    from realesrgan import RealESRGANer  # noqa: F401


def _initial_tile() -> int:
    override = os.environ.get("REALESRGAN_TILE")
    if override:
        value = int(override)
        if value <= 0:
            raise ValueError("REALESRGAN_TILE must be a positive integer")
        return value
    if not torch.cuda.is_available():
        return 128
    free_bytes, _ = torch.cuda.mem_get_info()
    free_gib = free_bytes / (1024 ** 3)
    if free_gib >= 5.0:
        return 1024
    if free_gib >= 3.5:
        return 768
    if free_gib >= 2.5:
        return 512
    if free_gib >= 1.7:
        return 384
    if free_gib >= 1.0:
        return 256
    if free_gib >= 0.65:
        return 192
    return 128


def _candidate_tiles(initial: int) -> list[int]:
    candidates = [initial]
    candidates.extend(tile for tile in TILE_CANDIDATES if tile < initial)
    return list(dict.fromkeys(candidates))


class ResidentRealESRGAN:
    """Load x2plus once and retain its weights between SAM3 operations."""

    def __init__(self) -> None:
        _verify_model()
        _install_torchvision_compatibility()
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=MODEL_SCALE,
        )
        self.tile = _initial_tile()
        self.upsampler = RealESRGANer(
            scale=MODEL_SCALE,
            model_path=str(MODEL_PATH),
            model=model,
            tile=self.tile,
            tile_pad=10,
            pre_pad=10,
            half=device.type == "cuda",
            device=device,
        )
        self.device = device
        self.lock = threading.RLock()
        print(
            f"RealESRGAN_x2plus resident on {device}; target long edge="
            f"{TARGET_LONG_EDGE}, initial tile={self.tile}",
            flush=True,
        )

    def _release_intermediates(self) -> None:
        # Keep self.upsampler.model resident; release only per-image tensors.
        self.upsampler.img = None
        self.upsampler.output = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def enhance(self, image: Image.Image) -> Image.Image:
        source_long_edge = max(image.size)
        if source_long_edge >= TARGET_LONG_EDGE:
            return image
        target_scale = TARGET_LONG_EDGE / source_long_edge
        source = np.asarray(image.convert("RGB"))
        source_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        last_error: Exception | None = None
        with self.lock:
            for tile in _candidate_tiles(self.tile):
                self.upsampler.tile_size = tile
                try:
                    # RealESRGANer prints one line per tile. Suppress that noisy
                    # implementation detail while retaining our OOM retry log.
                    with redirect_stdout(io.StringIO()):
                        output_bgr, _ = self.upsampler.enhance(
                            source_bgr, outscale=target_scale
                        )
                    self.tile = tile
                    self._release_intermediates()
                    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(output_rgb)
                except (torch.cuda.OutOfMemoryError, RuntimeError, UnboundLocalError) as exc:
                    if (
                        not isinstance(exc, (torch.cuda.OutOfMemoryError, UnboundLocalError))
                        and "out of memory" not in str(exc).lower()
                    ):
                        self._release_intermediates()
                        raise
                    last_error = exc
                    self._release_intermediates()
                    print(
                        f"Real-ESRGAN tile {tile} OOM; retrying with a smaller tile",
                        flush=True,
                    )
            raise torch.cuda.OutOfMemoryError(
                "Real-ESRGAN could not process the image even with tile 64"
            ) from last_error


_RESIDENT_MODEL: ResidentRealESRGAN | None = None
_MODEL_LOCK = threading.RLock()


def resident_super_resolution() -> ResidentRealESRGAN:
    global _RESIDENT_MODEL
    with _MODEL_LOCK:
        if _RESIDENT_MODEL is None:
            _RESIDENT_MODEL = ResidentRealESRGAN()
        return _RESIDENT_MODEL


def upscale_for_sam3(image: Image.Image) -> Image.Image:
    """Upscale toward a 2K long edge while retaining the model for later calls."""
    if not _enabled():
        return image
    return resident_super_resolution().enhance(image)


def status() -> dict:
    model = _RESIDENT_MODEL
    return {
        "enabled": _enabled(),
        "loaded": model is not None,
        "model": "RealESRGAN_x2plus",
        "model_scale": MODEL_SCALE,
        "target_long_edge": TARGET_LONG_EDGE,
        "tile": model.tile if model is not None else None,
        "device": str(model.device) if model is not None else None,
    }
