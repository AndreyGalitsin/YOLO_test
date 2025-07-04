# -*- coding: utf-8 -*-
"""
----------------------
Pipeline step 2:  «Object‑proposals → CLIP‑matching → pseudo‑labels»

Input
-----
1. --images_folder_path       : folder with *.jpg images (one level, any name)
2. --yolo_pred     : COCO JSON with YOLO‑25 predictions (step‑1 output)
3. --gallery_dir   : folder with 1‑3 *.jpg per class (file name == class name, any suffix)

Output
------
* pseudo_labels.json    – COCO‑style annotations where "source" = "clip", "confidence_clip" added
* active_zone.csv       – proposals with 0.22 < cos < 0.28 to be checked by human
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor
import clip
import matplotlib.pyplot as plt


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


def create_free_mask(img_shape: Tuple[int, int], boxes: List[List[int]]) -> np.ndarray:
    """Mask pixels *not* covered by any bbox (1 = free)."""
    H, W = img_shape
    mask = np.ones((H, W), np.uint8)  # 1 = free
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)  # 0 = occupied
    # dilate a bit to close gaps between bbox of same object
    # mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return mask


def connected_components(mask: np.ndarray, min_area: int = 3000) -> List[List[int]]:
    """Return bounding boxes (x1,y1,x2,y2) for each connected free region > min_area pixels."""
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    props = []
    for i in range(1, num):  # skip background label 0
        x, y, w, h, area = stats[i]
        if area >= min_area:
            props.append([x, y, x + w, y + h])
    return props


def main(images_folder_path, yolo_pred, gallery_dir, out_pseudo, out_active,
         high_tau: float = 0.8, low_tau: float = 0.5):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. load YOLO bbox mask map
    yolo_boxes = load_coco_boxes(Path(yolo_pred))

    # 2. load SAM‑B model
    sam = sam_model_registry["vit_b"](checkpoint="checkpoints/sam_vit_b_01ec64.pth")
    sam.to(device)
    predictor = SamPredictor(sam)

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

        # free mask
        boxes = yolo_boxes.get(str(img_path))
        boxes = []
        mask_free = create_free_mask((H, W), boxes)

        # connected free regions → proposals
        props = connected_components(mask_free)
        if not props:
            continue

        # run SAM encoder once per image
        predictor.set_image(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))

        for (x1, y1, x2, y2) in props:
            sam_box = np.array([x1, y1, x2, y2], dtype=np.float32)
            masks, _, _ = predictor.predict(box=sam_box[None, :], multimask_output=False)
            mask = masks[0].astype(np.uint8)
            # фильтр: крупные маски
            if mask.sum() / (H * W) > 0.40:
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            vis = im_bgr.copy()
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.title(f"mask contours: {img_path.name}")
            plt.axis("off")
            plt.show()

            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                if cw * ch < 1500:
                    continue

                bx1, by1, bx2, by2 = cx, cy, cx + cw, cy + ch
                pad = 4
                bx1 = max(bx1 - pad, 0)
                by1 = max(by1 - pad, 0)
                bx2 = min(bx2 + pad, W - 1)
                by2 = min(by2 + pad, H - 1)

                crop = im_bgr[by1:by2 + 1, bx1:bx2 + 1]
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

                plt.figure(figsize=(3, 3))
                plt.imshow(crop_pil)
                plt.title(f"{best_class}\n{best_sim:.2f}")
                plt.axis("off")
                plt.show()

                if best_sim >= high_tau:
                    pseudo_annotations.append({
                        "file_name": str(img_path),
                        "bbox": [int(bx1), int(by1), int(bx2 - bx1), int(by2 - by1)],
                        "class": best_class,
                        "confidence_clip": best_sim,
                    })
                elif best_sim >= low_tau:
                    active_rows.append({
                        "file_name": str(img_path),
                        "x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
                        "top_class": best_class,
                        "similarity": best_sim,
                    })




            # # convert to SAM box format (x1,y1,x2,y2) float32
            # sam_box = np.array([x1, y1, x2, y2], dtype=np.float32)
            # masks, _, _ = predictor.predict(box=sam_box[None, :], multimask_output=False)
            # mask = masks[0].astype(np.uint8)
            # # if mask mostly empty skip
            # if mask.sum() < 500:  # pixels
            #     continue
            # # bbox of mask for tighter crop
            # ys, xs = np.where(mask > 0)
            # if len(xs) == 0:
            #     continue
            # bx1, by1, bx2, by2 = xs.min(), ys.min(), xs.max(), ys.max()
            # pad = 4
            # bx1 = max(bx1 - pad, 0); by1 = max(by1 - pad, 0)
            # bx2 = min(bx2 + pad, W - 1); by2 = min(by2 + pad, H - 1)
            # crop = im_bgr[by1:by2 + 1, bx1:bx2 + 1]
            # crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            # # CLIP embedding
            # with torch.no_grad():
            #     emb = clip_model.encode_image(clip_preprocess(crop_pil).unsqueeze(0).to(device))
            # emb = F.normalize(emb, dim=-1).cpu().squeeze(0)
            # # similarity to gallery
            # cos_sim = torch.matmul(gallery_matrix, emb).detach().numpy()
            # best_idx = int(cos_sim.argmax())
            # best_sim = float(cos_sim[best_idx])
            # best_class = gallery_names[best_idx]
            #
            # if best_sim >= high_tau:
            #     pseudo_annotations.append({
            #         "file_name": str(img_path),
            #         "bbox": [int(bx1), int(by1), int(bx2 - bx1), int(by2 - by1)],
            #         "class": best_class,
            #         "confidence_clip": best_sim,
            #     })
            # elif best_sim >= low_tau:
            #     active_rows.append({
            #         "file_name": str(img_path),
            #         "x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
            #         "top_class": best_class,
            #         "similarity": best_sim,
            #     })


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
