import json
import pickle
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import cv2
from tqdm import tqdm
import open_clip
import csv
import matplotlib.pyplot as plt


class Labeling:
    def __init__(self):
        self.images_folder_path = Path("data/task_169")
        self.coco_annotations_json_path = Path("data/task_169/coco_annotations.json")
        self.to_review_csv_path = Path("data/task_169/to_review.csv")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.reference_embeddings = self.load_reference_embeddings("data/reference_embeddings.pkl")


        # YOLO
        self.yolo_model = YOLO("checkpoints/yolo_marking.pt")
        self.yolo_model.fuse()
        self.yolo_model.to(self.device).eval()
        self.yolo_th = 0.8

        # SAM
        sam = sam_model_registry["vit_b"](checkpoint="checkpoints/sam_vit_b_01ec64.pth")
        self.mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=12,
            pred_iou_thresh=0.85,
            min_mask_region_area=10000,
            stability_score_thresh=0.9
        )


        self.iou_suppression_th = 0.5

        # CLIP
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.clip_model = self.clip_model.to(self.device).eval()
        self.clip_th_strong = 0.7
        self.clip_th_weak = 0.5


    @staticmethod
    def load_reference_embeddings(reference_embeddings_path):
        with open(reference_embeddings_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def get_categories(self):
        yolo_categories = [{"id": key, "name": self.yolo_model.names[key]} for key in self.yolo_model.names]
        next_category_id = max(self.yolo_model.names.keys()) + 1
        additional_categories = [{"id": next_category_id+i, "name": cat_name} for i, cat_name in enumerate(self.reference_embeddings.keys())]
        return yolo_categories + additional_categories


    def main(self):
        coco_categories = self.get_categories()
        coco = {"images": [], "annotations": [], "categories": coco_categories}
        review_rows = []
        ann_id = 1
        img_id = 0

        for img_path in tqdm((sorted(self.images_folder_path.glob("*/*.jpg"))), desc="Processing images"):
            img_id += 1
            image = cv2.imread(str(img_path))
            h, w = image.shape[:2]
            coco["images"].append({"id": img_id, "file_name": str(img_path), "width": w, "height": h})

            yolo_res = self.yolo_model.predict(image, conf=self.yolo_th, device=self.device, verbose=False)[0]
            yolo_boxes = []

            for box, cls, score in zip(yolo_res.boxes.xyxy.cpu(), yolo_res.boxes.cls.cpu(), yolo_res.boxes.conf.cpu()):
                x1, y1, x2, y2 = box.tolist()
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                    "confidence": float(score)
                })
                yolo_boxes.append([x1, y1, x2, y2])
                ann_id += 1

            # Маскируем YOLO-объекты
            image_masked = image.copy()
            for x1, y1, x2, y2 in yolo_boxes:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                image_masked[y1:y2, x1:x2] = 0

            # SAM → маски
            masks = self.mask_generator.generate(image_masked)
            masks = [m for m in masks if m["area"] >= 10000]
            masks = sorted(masks, key=lambda m: m["area"], reverse=True)[:20]
            for mask in masks:
                x, y, w_box, h_box = mask["bbox"]
                x2, y2 = x + w_box, y + h_box
                mask_box = [x, y, x2, y2]

                # Скипаем, если перекрывается с YOLO
                if any(self.compute_iou(mask_box, ybox) > self.iou_suppression_th for ybox in yolo_boxes):
                    continue

                # Вырезаем объект по bbox
                obj_crop = image[y:y2, x:x2]
                if obj_crop.size == 0:
                    continue

                # CLIP эмбеддинг
                img_pil = Image.fromarray(cv2.cvtColor(obj_crop, cv2.COLOR_BGR2RGB))
                clip_input = self.clip_preprocess(img_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    obj_emb = self.clip_model.encode_image(clip_input)
                    obj_emb /= obj_emb.norm(dim=-1, keepdim=True)
                    # obj_emb = obj_emb.squeeze(0).cpu().numpy()
                    obj_emb = obj_emb.squeeze(0)


                # Сравнение с эталонами
                best_class = None
                best_sim = -1
                for class_name, emb_list in self.reference_embeddings.items():
                    gallery = torch.tensor(emb_list).to(self.device)  # [N, D]
                    sim = torch.matmul(gallery, obj_emb)
                    max_sim = sim.max().item()
                    # sims = [np.dot(obj_emb, ref_emb) for ref_emb in emb_list]
                    # max_sim = max(sims)
                    if max_sim > best_sim:
                        best_sim = max_sim
                        best_class = class_name


                if best_sim >= self.clip_th_strong:
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": next(el['id'] for el in coco_categories if el['name'] == best_class),
                        "bbox": [x, y, w_box, h_box],
                        "area": w_box * h_box,
                        "iscrowd": 0,
                        "confidence": float(best_sim)
                    })
                    ann_id += 1

                    plt.figure(figsize=(15, 8))
                    plt.imshow(img_pil)
                    plt.title(f"{img_path.name}, {best_class}\n{best_sim:.2f}")
                    plt.axis("off")
                    plt.show()

                elif best_sim >= self.clip_th_weak:
                    review_rows.append({
                        "image_path": str(img_path),
                        "bbox": [x, y, w_box, h_box],
                        "suggested_class": best_class,
                        "similarity": best_sim
                    })


        with open(self.coco_annotations_json_path, "w") as f:
            json.dump(coco, f, indent=2)

        with open(self.to_review_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "bbox", "suggested_class", "similarity"])
            writer.writeheader()
            for row in review_rows:
                writer.writerow(row)

        print(f"\nSaved {len(coco['annotations'])} coco_annotations and {len(review_rows)} uncertain detections")


if __name__ == "__main__":
    lb = Labeling()
    lb.main()
