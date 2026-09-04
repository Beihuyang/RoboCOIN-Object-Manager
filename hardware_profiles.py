"""Runtime profiles for local development and GPU-only server inference."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    sam3_official_prompt_batch: bool
    sam3_prompt_batch_size: int
    sam3_mask_upsample_chunk_size: int
    sam3_offload_masks_to_cpu: bool
    tracker_offload_video_to_cpu: bool
    tracker_offload_state_to_cpu: bool
    tracker_async_loading_frames: bool
    qwen_batch_size: int
    clip_batch_size: int


PROFILES = {
    "local": HardwareProfile(
        name="local",
        sam3_official_prompt_batch=False,
        sam3_prompt_batch_size=1,
        sam3_mask_upsample_chunk_size=8,
        sam3_offload_masks_to_cpu=True,
        tracker_offload_video_to_cpu=True,
        tracker_offload_state_to_cpu=True,
        tracker_async_loading_frames=False,
        qwen_batch_size=2,
        clip_batch_size=16,
    ),
    "a800": HardwareProfile(
        name="a800",
        sam3_official_prompt_batch=True,
        sam3_prompt_batch_size=16,
        sam3_mask_upsample_chunk_size=64,
        sam3_offload_masks_to_cpu=False,
        tracker_offload_video_to_cpu=False,
        tracker_offload_state_to_cpu=False,
        tracker_async_loading_frames=False,
        qwen_batch_size=8,
        clip_batch_size=64,
    ),
}


def default_profile_name() -> str:
    return os.environ.get("ROBOCOIN_HARDWARE_PROFILE", "local")


def get_profile(name: str | None = None) -> HardwareProfile:
    resolved = name or default_profile_name()
    try:
        return PROFILES[resolved]
    except KeyError as exc:
        raise ValueError(
            f"Unknown hardware profile {resolved!r}; choose one of {sorted(PROFILES)}"
        ) from exc
