"""Official SAM3 API batching: one image with several independent text queries."""

from __future__ import annotations

from PIL import Image
import torch


def infer_text_queries(
    model,
    image: Image.Image,
    prompts: list[str],
    *,
    original_size: tuple[int, int],
    threshold: float,
) -> dict[str, dict]:
    """Run SAM3's official Datapoint/collator API and return one result per query."""
    from sam3.eval.postprocessors import PostProcessImage
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        FindQueryLoaded,
        Image as SAMImage,
        InferenceMetadata,
    )
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI,
        NormalizeAPI,
        RandomResizeAPI,
        ToTensorAPI,
    )

    width, height = image.size
    original_width, original_height = original_size
    datapoint = Datapoint(find_queries=[], images=[
        SAMImage(data=image, objects=[], size=[height, width])
    ])
    query_ids: dict[str, int] = {}
    for query_id, prompt in enumerate(prompts, start=1):
        query_ids[prompt] = query_id
        datapoint.find_queries.append(FindQueryLoaded(
            query_text=prompt,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=query_id,
                original_image_id=query_id,
                original_category_id=1,
                original_size=[original_width, original_height],
                object_id=0,
                frame_index=0,
            ),
        ))
    transform = ComposeAPI(transforms=[
        RandomResizeAPI(
            sizes=1008, max_size=1008, square=True, consistent_transform=False
        ),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    batch = collate([transform(datapoint)], dict_key="dummy")["dummy"]
    device = next(model.parameters()).device
    batch = copy_data_to_device(batch, device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, cache_enabled=False
    ):
        output = model(batch)
    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=max(0.0, threshold - 1e-7),
        to_cpu=False,
    )
    processed = postprocessor.process_results(output, batch.find_metadatas)
    return {prompt: processed[query_id] for prompt, query_id in query_ids.items()}
