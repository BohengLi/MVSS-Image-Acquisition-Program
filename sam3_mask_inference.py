from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image


def _split_prompts(text: str) -> list[str]:
    return [item.strip() for item in str(text).replace("\uFF0C", ",").split(",") if item.strip()]


def _find_cached_checkpoint(sam3_root: Path, filename: str = "sam3.pt") -> Path | None:
    candidates: list[Path] = []
    for env_name in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "hub")
    candidates.append(sam3_root / "hf_cache" / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

    seen: set[str] = set()
    for cache_root in candidates:
        key = str(cache_root).casefold()
        if key in seen:
            continue
        seen.add(key)
        snapshots = cache_root / "models--facebook--sam3" / "snapshots"
        if not snapshots.is_dir():
            continue
        for snapshot in sorted(snapshots.iterdir(), key=lambda item: item.name):
            checkpoint = snapshot / filename
            if checkpoint.is_file() and checkpoint.stat().st_size > 0:
                return checkpoint
    return None


def _select_masks(masks, boxes, scores, labels, top_k: int, selection: str):
    import torch

    if scores.numel() == 0:
        return masks, boxes, scores, labels

    order = torch.argsort(scores.float(), descending=True)
    if top_k > 0:
        order = order[:top_k]

    selection = str(selection or "union").strip().lower()
    if selection == "best":
        order = order[:1]
    elif selection == "largest":
        flat = masks.detach().reshape(masks.shape[0], -1).float()
        largest_index = int(torch.argmax(flat.sum(dim=1)).item())
        order = torch.tensor([largest_index], device=scores.device)

    selected_labels = [labels[int(index)] for index in order.detach().cpu().tolist()]
    return masks[order], boxes[order], scores[order], selected_labels


def _write_empty_semantic_outputs(
    image: Image.Image,
    output_instance_map: Path | None,
    output_semantic_map: Path | None,
    output_confidence_map: Path | None,
    output_label_map: Path | None,
) -> None:
    shape = (image.height, image.width)
    if output_instance_map is not None:
        output_instance_map.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_instance_map, np.zeros(shape, dtype=np.int32))
    if output_semantic_map is not None:
        output_semantic_map.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_semantic_map, np.zeros(shape, dtype=np.int32))
    if output_confidence_map is not None:
        output_confidence_map.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_confidence_map, np.zeros(shape, dtype=np.float32))
    if output_label_map is not None:
        output_label_map.parent.mkdir(parents=True, exist_ok=True)
        output_label_map.write_text(
            json.dumps(
                {
                    "labels": [{"semantic_id": 0, "label": "background"}],
                    "instances": [],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _empty_result(
    image: Image.Image,
    output_mask: Path,
    output_json: Path,
    prompts: list[str],
    status: str,
    output_instance_map: Path | None = None,
    output_semantic_map: Path | None = None,
    output_confidence_map: Path | None = None,
    output_label_map: Path | None = None,
) -> None:
    mask = Image.new("L", image.size, 0)
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_mask)
    _write_empty_semantic_outputs(
        image,
        output_instance_map,
        output_semantic_map,
        output_confidence_map,
        output_label_map,
    )
    output_json.write_text(
        json.dumps(
            {
                "status": status,
                "prompts": prompts,
                "object_count": 0,
                "mask_pixels": 0,
                "mask_ratio": 0.0,
                "semantic_labels": [{"semantic_id": 0, "label": "background"}],
                "objects": [],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 image segmentation and write one combined object mask.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-mask", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-instance-map", default="")
    parser.add_argument("--output-semantic-map", default="")
    parser.add_argument("--output-confidence-map", default="")
    parser.add_argument("--output-label-map", default="")
    parser.add_argument("--sam3-root", default=r"D:\SAM3")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--prompt", default="object")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--selection", default="union", choices=("union", "best", "largest"))
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = perf_counter()

    image_path = Path(args.image)
    output_mask = Path(args.output_mask)
    output_json = Path(args.output_json)
    output_instance_map = Path(args.output_instance_map) if args.output_instance_map else None
    output_semantic_map = Path(args.output_semantic_map) if args.output_semantic_map else None
    output_confidence_map = Path(args.output_confidence_map) if args.output_confidence_map else None
    output_label_map = Path(args.output_label_map) if args.output_label_map else None
    sam3_root = Path(args.sam3_root)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not sam3_root.is_dir():
        raise FileNotFoundError(f"SAM3 root not found: {sam3_root}")

    sys.path.insert(0, str(sam3_root))
    os.environ.setdefault("HF_HOME", str(sam3_root / "hf_cache"))
    os.environ.setdefault("HF_HUB_CACHE", str(sam3_root / "hf_cache" / "hub"))

    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    image = Image.open(image_path).convert("RGB")
    prompts = _split_prompts(args.prompt)
    if not prompts:
        _empty_result(
            image,
            output_mask,
            output_json,
            prompts,
            "empty_prompt",
            output_instance_map,
            output_semantic_map,
            output_confidence_map,
            output_label_map,
        )
        return 0

    checkpoint = Path(args.checkpoint) if args.checkpoint else _find_cached_checkpoint(sam3_root)
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for SAM3 but torch.cuda.is_available() is false.")

    model = build_sam3_image_model(
        checkpoint_path=None if checkpoint is None else str(checkpoint),
        image_size=int(args.resolution),
    )
    model = model.to(device).eval()
    processor = Sam3Processor(
        model,
        resolution=int(args.resolution),
        device=device,
        confidence_threshold=float(args.threshold),
    )

    all_masks = []
    all_boxes = []
    all_scores = []
    all_labels: list[str] = []
    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with torch.inference_mode(), autocast:
        state = processor.set_image(image)
        for prompt in prompts:
            output = processor.set_text_prompt(state=state, prompt=prompt)
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"].float()
            keep = scores >= float(args.threshold)
            if keep.numel() > 0:
                masks = masks[keep]
                boxes = boxes[keep]
                scores = scores[keep]
            if scores.numel() == 0:
                continue
            all_masks.append(masks)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.extend([prompt] * int(scores.numel()))

    if not all_scores:
        _empty_result(
            image,
            output_mask,
            output_json,
            prompts,
            "no_masks",
            output_instance_map,
            output_semantic_map,
            output_confidence_map,
            output_label_map,
        )
        return 0

    masks = torch.cat(all_masks, dim=0)
    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    masks, boxes, scores, selected_labels = _select_masks(
        masks,
        boxes,
        scores,
        all_labels,
        max(0, int(args.top_k)),
        args.selection,
    )

    masks_cpu = masks.detach().squeeze(1).cpu().numpy().astype(bool)
    boxes_cpu = boxes.detach().float().cpu().numpy()
    scores_cpu = scores.detach().float().cpu().numpy()

    semantic_labels = [{"semantic_id": 0, "label": "background"}]
    label_to_id: dict[str, int] = {"background": 0}
    for label in selected_labels:
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id)
            semantic_labels.append({"semantic_id": label_to_id[label], "label": label})

    instance_map = np.zeros((image.height, image.width), dtype=np.int32)
    semantic_map = np.zeros((image.height, image.width), dtype=np.int32)
    confidence_map = np.zeros((image.height, image.width), dtype=np.float32)
    if masks_cpu.size:
        # Low-confidence masks are assigned first so stronger overlapping masks win.
        for mask_index in np.argsort(scores_cpu):
            mask = masks_cpu[int(mask_index)]
            if mask.shape != (image.height, image.width):
                mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
                mask_img = mask_img.resize(image.size, Image.Resampling.NEAREST)
                mask = np.asarray(mask_img) > 0
            instance_id = int(mask_index) + 1
            label = selected_labels[int(mask_index)]
            semantic_id = int(label_to_id[label])
            instance_map[mask] = instance_id
            semantic_map[mask] = semantic_id
            confidence_map[mask] = float(scores_cpu[int(mask_index)])

    combined = instance_map > 0
    if combined.shape != (image.height, image.width):
        combined_img = Image.fromarray((combined.astype(np.uint8) * 255), mode="L")
        combined_img = combined_img.resize(image.size, Image.Resampling.NEAREST)
        combined = np.asarray(combined_img) > 0

    output_mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((combined.astype(np.uint8) * 255), mode="L").save(output_mask)
    if output_instance_map is not None:
        output_instance_map.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_instance_map, instance_map.astype(np.int32))
    if output_semantic_map is not None:
        output_semantic_map.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_semantic_map, semantic_map.astype(np.int32))
    if output_confidence_map is not None:
        output_confidence_map.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_confidence_map, confidence_map.astype(np.float32))

    objects = []
    for index, (mask, box, score, label) in enumerate(zip(masks_cpu, boxes_cpu, scores_cpu, selected_labels), start=1):
        semantic_id = int(label_to_id[label])
        objects.append(
            {
                "index": index,
                "instance_id": index,
                "label": label,
                "semantic_id": semantic_id,
                "score": round(float(score), 6),
                "box_xyxy": [round(float(value), 2) for value in box.tolist()],
                "area_px": int(np.count_nonzero(mask)),
                "assigned_pixels": int(np.count_nonzero(instance_map == index)),
            }
        )

    if output_label_map is not None:
        output_label_map.parent.mkdir(parents=True, exist_ok=True)
        output_label_map.write_text(
            json.dumps(
                {
                    "labels": semantic_labels,
                    "instances": [
                        {
                            "instance_id": item["instance_id"],
                            "semantic_id": item["semantic_id"],
                            "label": item["label"],
                            "score": item["score"],
                        }
                        for item in objects
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    payload = {
        "status": "ok",
        "prompts": prompts,
        "threshold": float(args.threshold),
        "top_k": int(args.top_k),
        "selection": args.selection,
        "resolution": int(args.resolution),
        "device": device,
        "checkpoint": "" if checkpoint is None else str(checkpoint),
        "object_count": len(objects),
        "mask_pixels": int(np.count_nonzero(combined)),
        "mask_ratio": float(np.count_nonzero(combined) / max(combined.size, 1)),
        "semantic_labels": semantic_labels,
        "semantic_map": "" if output_semantic_map is None else str(output_semantic_map),
        "instance_map": "" if output_instance_map is None else str(output_instance_map),
        "confidence_map": "" if output_confidence_map is None else str(output_confidence_map),
        "label_map": "" if output_label_map is None else str(output_label_map),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "objects": objects,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
