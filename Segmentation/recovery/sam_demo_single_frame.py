"""
Single-frame segmentation demo using GroundingDINO (text prompts -> boxes) + SAM (boxes -> masks).

Outputs an overlay image highlighting the dumpster and garbage bag.
"""
from __future__ import annotations

import colorsys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor, pipeline


def box_iou_xyxy(box_a, box_b) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, (xa2 - xa1)) * max(0.0, (ya2 - ya1))
    area_b = max(0.0, (xb2 - xb1)) * max(0.0, (yb2 - yb1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_per_label(dets, iou_thresh: float = 0.5):
    kept = []
    for det in dets:
        box = det["box"]
        bbox = [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
        overlap = False
        for k in kept:
            kbox = k["box"]
            if box_iou_xyxy(bbox, [kbox["xmin"], kbox["ymin"], kbox["xmax"], kbox["ymax"]]) > iou_thresh:
                overlap = True
                break
        if not overlap:
            kept.append(det)
    return kept


def _pick_colors(n: int) -> List[Tuple[int, int, int]]:
    # Evenly spaced hues for consistent visualization
    return [
        tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / max(n, 1), 0.65, 1.0))
        for i in range(n)
    ]


def _resize_with_limit(img: np.ndarray, max_side: int = 1280) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale == 1.0:
        return img, 1.0
    new_size = (int(w * scale), int(h * scale))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def detect_boxes_zero_shot(
    image_rgb: np.ndarray,
    labels: List[str],
    device: torch.device,
    score_threshold: float = 0.35,
):
    pil_image = Image.fromarray(image_rgb)
    detector = pipeline(
        task="zero-shot-object-detection",
        model="IDEA-Research/grounding-dino-base",
        device=0 if device.type == "cuda" else -1,
    )
    results = detector(pil_image, candidate_labels=labels, threshold=score_threshold)
    # Ensure consistent ordering by label priority then score
    results.sort(key=lambda r: (-r["score"], labels.index(r["label"]) if r["label"] in labels else 999))
    return results


def run_segmentation(
    image_bgr: np.ndarray,
    labels: List[str],
    device: torch.device,
):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    dets = detect_boxes_zero_shot(image_rgb, labels, device=device)
    if len(dets) == 0:
        return image_bgr, [], {
            "fill_ratio": 0.0,
            "fill_percent": 0.0,
            "fill_level": "LOW",
            "container_area_px": 0,
            "trash_area_px": 0,
        }

    # NMS per label
    filtered = []
    for lbl in labels:
        per_label = [d for d in dets if d["label"] == lbl]
        kept = nms_per_label(per_label, iou_thresh=0.5)
        filtered.extend(kept)

    # Geometry filters
    img_h, img_w = image_bgr.shape[:2]
    img_area = float(img_h * img_w)
    dumpsters_all = [d for d in filtered if d["label"] == labels[0]]

    dumpsters = [
        d
        for d in dumpsters_all
        if d["box"]["ymin"] >= img_h * 0.25
        and (d["box"]["xmax"] - d["box"]["xmin"]) * (d["box"]["ymax"] - d["box"]["ymin"]) >= img_area * 0.05
    ]
    dumpsters.sort(key=lambda d: -d["score"])
    if not dumpsters and dumpsters_all:
        dumpsters = [dumpsters_all[0]]  # fallback to top-scoring dumpster
    dumpster_ref = dumpsters[0] if dumpsters else None

    if dumpster_ref:
        dxmin, dymin, dxmax, dymax = (
            dumpster_ref["box"]["xmin"],
            dumpster_ref["box"]["ymin"],
            dumpster_ref["box"]["xmax"],
            dumpster_ref["box"]["ymax"],
        )
        d_area = (dxmax - dxmin) * (dymax - dymin)
        d_xcenter = 0.5 * (dxmin + dxmax)

    filtered = []
    if dumpster_ref:
        filtered.append(dumpster_ref)

    if len(filtered) == 0:
        return image_bgr, [], {
            "fill_ratio": 0.0,
            "fill_percent": 0.0,
            "fill_level": "LOW",
            "container_area_px": 0,
            "trash_area_px": 0,
        }

    boxes = []
    ordered_labels = []
    d_area = None
    if dumpster_ref:
        dxmin, dymin, dxmax, dymax = (
            dumpster_ref["box"]["xmin"],
            dumpster_ref["box"]["ymin"],
            dumpster_ref["box"]["xmax"],
            dumpster_ref["box"]["ymax"],
        )
        d_area = (dxmax - dxmin) * (dymax - dymin)

    for det in filtered:
        box = det["box"]
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        boxes.append([xmin, ymin, xmax, ymax])
        ordered_labels.append(det["label"])

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base", torch_dtype=torch_dtype).to(device)
    sam_model.eval()

    input_boxes = [np.array(boxes, dtype=np.float32).tolist()]
    inputs = sam_processor(
        images=Image.fromarray(image_rgb),
        input_boxes=input_boxes,
        return_tensors="pt",
    )
    processed_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            if v.dtype.is_floating_point:
                processed_inputs[k] = v.to(device=device, dtype=sam_model.dtype)
            else:
                processed_inputs[k] = v.to(device=device)
        else:
            processed_inputs[k] = v
    inputs = processed_inputs

    with torch.inference_mode():
        outputs = sam_model(**inputs)

    masks = sam_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
    )[0]

    overlay = draw_masks_overlay(image_bgr, masks, ordered_labels)
    stats = compute_fill_stats(masks, ordered_labels)
    return overlay, list(zip(ordered_labels, filtered)), stats


def draw_masks_overlay(
    image_bgr: np.ndarray,
    masks: torch.Tensor,
    labels: List[str],
    alpha: float = 0.45,
) -> np.ndarray:
    overlay = image_bgr.copy()
    colors = _pick_colors(len(labels))
    for idx, (mask, label) in enumerate(zip(masks, labels)):
        mask_np = np.array(mask.cpu())
        mask_np = np.squeeze(mask_np)
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        if mask_np.ndim != 2:
            raise ValueError(f"Unexpected mask shape: {mask_np.shape}")
        mask_np = mask_np.astype(bool)
        color = colors[idx % len(colors)]
        overlay[mask_np] = (
            overlay[mask_np] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
        )

        # Draw label near the top-left of the mask
        ys, xs = np.where(mask_np)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        cv2.putText(
            overlay,
            label,
            (x0, y0 - 5 if y0 > 10 else y0 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def compute_fill_stats(masks: torch.Tensor, labels: List[str]):
    label_masks = {}
    for mask, label in zip(masks, labels):
        mask_np = np.array(mask.cpu())
        mask_np = np.squeeze(mask_np)
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        mask_np = mask_np.astype(bool)
        label_masks.setdefault(label, []).append(mask_np)

    container_mask = None
    if "green metal dumpster on wheels" in label_masks:
        container_mask = np.logical_or.reduce(label_masks["green metal dumpster on wheels"])
    trash_mask = None
    if "black plastic trash bag" in label_masks:
        trash_mask = np.logical_or.reduce(label_masks["black plastic trash bag"])

    container_area = int(container_mask.sum()) if container_mask is not None else 0
    trash_area = int(trash_mask.sum()) if trash_mask is not None else 0
    fill_ratio = float(trash_area / container_area) if container_area > 0 else 0.0

    if fill_ratio < 0.15:
        fill_level = "LOW"
    elif fill_ratio < 0.45:
        fill_level = "MEDIUM"
    else:
        fill_level = "HIGH"

    return {
        "fill_ratio": round(fill_ratio, 4),
        "fill_percent": round(fill_ratio * 100, 2),
        "fill_level": fill_level,
        "container_area_px": container_area,
        "trash_area_px": trash_area,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    input_path = repo_root / "frame_at_3s.jpg"
    output_path = repo_root / "frame_at_3s_overlay.jpg"
    stats_path = repo_root / "single_frame_stats.json"

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input frame: {input_path}")

    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {input_path}")

    image_bgr, scale = _resize_with_limit(image_bgr, max_side=1280)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = ["green metal dumpster on wheels"]
    overlay, dets, stats = run_segmentation(image_bgr, labels=labels, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"Failed to save overlay to: {output_path}")

    # persist stats
    import json

    payload = {
        "frame_source": input_path.name,
        "timestamp_sec": 3.0,
        "detections": [
            {
                "label": lbl,
                "score": det["score"],
                "box": det["box"],
            }
            for lbl, det in dets
        ],
        "fill": stats,
    }
    stats_path.write_text(json.dumps(payload, indent=2))

    print(f"Saved overlay to {output_path} using device={device}")
    if dets:
        print("Detections (label, score, box):")
        for label, det in dets:
            box = det["box"]
            print(
                f"  {label}: score={det['score']:.3f}, box=({box['xmin']:.1f}, {box['ymin']:.1f}, "
                f"{box['xmax']:.1f}, {box['ymax']:.1f})"
            )
    else:
        print("No detections found for the provided prompts.")

    print(
        f"Fill ratio: {stats['fill_ratio']:.3f} ({stats['fill_percent']:.1f}%), level={stats['fill_level']}, "
        f"container_px={stats['container_area_px']}, trash_px={stats['trash_area_px']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
