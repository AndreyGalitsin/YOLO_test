import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import clip

def load_coco_boxes(json_path: Path) -> Dict[str, List[List[int]]]:
    """Return dict filename -> list[bbox xyxy]
    bbox stored as [x, y, w, h] (COCO), convert to xyxy int.
    """
    with open(json_path) as f:
        coco = json.load(f)
    img_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    boxes = {name: [] for name in img_id_to_name.values()}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        boxes[img_id_to_name[ann["image_id"]]].append([x1, y1, x2, y2])
    return boxes

def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return int(x1), int(y1), int(x2), int(y2)


def main(images_folder_path, yolo_pred, gallery_dir, out_pseudo, out_active,
         high_tau: float = 0.8, low_tau: float = 0.5):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. load YOLO bbox mask map
    yolo_boxes = load_coco_boxes(Path(yolo_pred))

    # 2. SAM AutoMask generator
    sam = sam_model_registry["vit_b"](checkpoint="checkpoints/sam_vit_b_01ec64.pth").to(device)
    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        min_mask_region_area=500,
    )

    # 3. load CLIP ViT‑B/32
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # 4. precompute gallery embeddings
    gallery_embeddings, gallery_names = [], []
    for img_path in sorted(Path(gallery_dir).glob("*/*.jpg")):
        class_name = str(img_path.parent.name)
        im = Image.open(img_path).convert("RGB")
        emb = clip_model.encode_image(clip_preprocess(im).unsqueeze(0).to(device))
        gallery_embeddings.append(emb.squeeze(0).cpu())
        gallery_names.append(class_name)
    gallery_matrix = F.normalize(torch.stack(gallery_embeddings), dim=-1)
    print(f"Loaded {len(gallery_names)} reference images from gallery_dir")

    pseudo_annotations = []
    active_rows = []

    # iterate images
    for img_path in tqdm(sorted(Path(images_folder_path).glob("*/*.jpg")), desc="images"):
        im_bgr = cv2.imread(str(img_path))
        H, W = im_bgr.shape[:2]

        occupied = np.zeros((H, W), np.uint8)
        for x1, y1, x2, y2 in yolo_boxes.get(img_path.name, []):
            cv2.rectangle(occupied, (x1, y1), (x2, y2), 255, -1)

        # SAM masks
        masks = mask_gen.generate(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))

        for m in masks:
            mask = m["segmentation"].astype(np.uint8)
            # если сильное перекрытие с занятыми → пропускаем
            inter = (mask & (occupied > 0)).sum() / mask.sum()
            if inter > 0.3:
                continue
            # площадь и aspect‑ratio фильтры
            if not 800 < mask.sum() < 0.30 * H * W:
                continue
            x1, y1, x2, y2 = bbox_from_mask(mask)
            if max((x2 - x1) / (y2 - y1 + 1e-6), (y2 - y1) / (x2 - x1 + 1e-6)) > 3:
                continue

            crop = im_bgr[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            # CLIP embedding
            with torch.no_grad():
                emb = clip_model.encode_image(clip_preprocess(crop_pil).unsqueeze(0).to(device))
            emb = F.normalize(emb, dim=-1).cpu().squeeze(0)

            # similarity to gallery
            cos_sim = torch.matmul(gallery_matrix, emb).detach().numpy()
            best_idx = int(cos_sim.argmax())
            best_sim = float(cos_sim[best_idx])
            best_class = gallery_names[best_idx]

            zone = None
            top2 = torch.topk(cos_sim, 2).values
            if (top2[0] - top2[1]).item() < 0.05:
                zone = "active"
            elif cos_sim >= high_tau:
                zone = "pseudo"
            elif cos_sim >= low_tau:
                zone = "active"
            else:
                continue

            if best_sim >= high_tau:
                pseudo_annotations.append({
                    "file_name": str(img_path),
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "class": best_class,
                    "confidence_clip": best_sim,
                })

            elif best_sim >= low_tau:
                active_rows.append({
                    "file_name": str(img_path),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "top_class": best_class,
                    "similarity": best_sim,
                })
    # save outputs
    with open(Path(out_pseudo), "w") as f:
        json.dump(pseudo_annotations, f, indent=2)
    pd.DataFrame(active_rows).to_csv(Path(out_active), index=False)
    print(f"Pseudo‑labels: {len(pseudo_annotations)}, Active rows: {len(active_rows)}")



if __name__ == "__main__":
    main(images_folder_path='data/task_169',
         yolo_pred="data/task_169/yolo_pred.json",
         gallery_dir='data/new_classes',
         out_pseudo="data/task_169/pseudo_labels.json",
         out_active="data/task_169/active_zone.csv")
