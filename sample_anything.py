import cv2
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import FastSAM


class SampleAnything:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FastSAM("checkpoints/FastSAM-x.pt")

    @staticmethod
    def visualization_polygon(img_bgr, masks):
        overlay = img_bgr.copy()
        rng = random.Random(0)
        for m in masks:
            for cnt in m["contours"]:
                if cnt.ndim != 2 or cnt.shape[0] < 3:
                    continue
                cnt_i32 = np.asarray(cnt, dtype=np.int32).reshape(-1, 1, 2)
                color = tuple(int(x) for x in rng.sample(range(50, 255), 3))
                cv2.polylines(overlay, [cnt_i32], True, color, 2)

        vis = cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"{len(masks)} masks (polygons)")
        plt.show()

    @staticmethod
    def visualization_bbox(img_bgr, bboxes):
        overlay = img_bgr.copy()
        rng = random.Random(42)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            color = tuple(int(x) for x in rng.sample(range(50, 255), 3))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        vis = cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"{len(bboxes)} boxes (bbox)")
        plt.show()

    def get_masked_crops(self, img_bgr, masks):
        crops = []
        for m in masks:
            mask = m["mask"]
            bbox = self.get_bbox_from_mask(mask)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            crop_img = img_bgr[y1:y2 + 1, x1:x2 + 1].copy()

            crop_mask = mask[y1:y2 + 1, x1:x2 + 1].astype(np.uint8)
            crop_mask = cv2.merge([crop_mask] * 3)  # → shape H×W×3

            masked = cv2.bitwise_and(crop_img, crop_mask * 255)
            crops.append(masked)

        return crops

    @staticmethod
    def get_max_area_mask(masks):
        return max(masks, key=lambda m: m["area"])

    @staticmethod
    def get_bbox_from_mask(mask: np.ndarray):
        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            return None
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        return (x1, y1, x2, y2)

    def inference(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        frame_area = H * W

        results = self.model.predict(
            img_rgb,
            device=self.device,
            conf=0.4,
            iou=0.9,
            retina_masks=True
        )
        r = results[0].cpu()

        masks_raw = []
        for idx, mask_tensor in enumerate(r.masks.data):
            mask_np = mask_tensor.bool().numpy()
            area = int(mask_np.sum())
            cnts, _ = cv2.findContours(mask_np.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            contours = [c.squeeze() for c in cnts if len(c) >= 3]
            if contours:
                masks_raw.append({
                    "mask": mask_np,
                    "contours": contours,
                    "area": area
                })


        # Разделение мультиобъектов на подмаски
        min_pixels = int(0.002 * frame_area)
        masks = []
        for m in masks_raw:
            for contour in m["contours"]:
                if contour.ndim != 2 or contour.shape[0] < 3:
                    continue

                mask_one = np.zeros_like(m["mask"], dtype=np.uint8)
                cv2.drawContours(mask_one, [contour.reshape(-1, 1, 2)], -1, 1, thickness=-1)
                area = int(mask_one.sum())

                if area >= min_pixels:
                    masks.append({
                        "mask": mask_one.astype(bool),
                        "contours": [contour],
                        "area": area
                    })

        # Удаление вложенных масок
        filtered = []
        for i, m1 in enumerate(masks):
            keep = True
            for j, m2 in enumerate(masks):
                if i == j:
                    continue

                if m1["area"] < 0.5 * m2["area"]:
                    inter = np.logical_and(m1["mask"], m2["mask"]).sum()
                    r1 = inter / m1["area"]
                    if r1 > 0.8:
                        keep = False

            if keep:
                filtered.append(m1)

        # print(f"Масок после фильтрации: {len(filtered)}, до фильтрации {len(masks)}")
        return filtered
