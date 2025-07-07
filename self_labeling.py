import json
import pickle
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import open_clip
import torch.nn.functional as F
from transformers import AutoImageProcessor, Dinov2Model
from torchvision import transforms

from yolo_detection import YOLODetection
from sample_anything import SampleAnything


class SelfLabeling:
    def __init__(self, emb_backbone="dinov2"):
        self.backbone = emb_backbone
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sa = SampleAnything()
        self.yd = YOLODetection(conf_th=0.8)

        if self.backbone == "clip":
            self.emb_model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self.emb_model = self.emb_model.to(self.device).eval()

        elif self.backbone == "dinov2":
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            self.emb_model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(self.device).eval()

        else:
            raise ValueError("Unsupported backbone. Choose 'clip' or 'dinov2'")

        # Augmentations
        self.augment = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.RandomRotation(15),
                transforms.GaussianBlur(kernel_size=(3, 3))
            ], p=0.7),
            transforms.Resize((224, 224))
        ])


    def extract_embedding(self, img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        if self.backbone == "clip":
            img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.emb_model.encode_image(img_tensor)

        elif self.backbone == "dinov2":
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.emb_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1)


        embedding = F.normalize(embedding, dim=-1)
        return embedding.squeeze(0)

    def create_embedding_matrix(self, data_dir: Path, pkl_path: Path=Path('data/new_classes/embedding_matrix.pkl')) -> dict[str, torch.Tensor]:
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                print('embedding_matrix загружена из pkl')
                return pickle.load(f)

        class_embeddings = defaultdict(list)
        for img_path in tqdm(sorted(data_dir.glob("*/*.jpg")), desc="images"):
            class_name = img_path.parent.name
            img_bgr = cv2.imread(str(img_path))

            masks = self.sa.inference(img_bgr)
            max_area_mask = self.sa.get_max_area_mask(masks)
            crops, _, _ = self.sa.get_masked_crops(img_bgr, [max_area_mask])
            assert len(crops) == 1
            crop_bgr = crops[0]

            emb = self.extract_embedding(crop_bgr)
            class_embeddings[class_name].append(emb)

            pil_crop = Image.fromarray(crop_bgr)
            for _ in range(4):
                aug_bgr = self.augment(pil_crop)
                aug_bgr = np.array(aug_bgr)
                emb_aug = self.extract_embedding(aug_bgr)
                class_embeddings[class_name].append(emb_aug)

        embedding_matrix = {
            cls: F.normalize(torch.stack(embs).mean(dim=0), dim=-1)
            for cls, embs in class_embeddings.items()
            if embs
        }

        with open(pkl_path, "wb") as f:
            pickle.dump(embedding_matrix, f)
            print(f"Сохранено в {pkl_path}")

        return embedding_matrix

    def compare(self, crop_bgr: np.ndarray, embedding_matrix: dict[str, torch.Tensor]):
        class_names = list(embedding_matrix.keys())
        class_matrix = torch.stack([embedding_matrix[name] for name in class_names]).to(self.device)
        class_matrix = F.normalize(class_matrix, dim=-1)

        emb = self.extract_embedding(crop_bgr).to(self.device)
        emb = emb.unsqueeze(0)

        cos_sim = F.cosine_similarity(emb, class_matrix.unsqueeze(0), dim=-1)[0]

        best_idx = torch.argmax(cos_sim).item()
        best_score = cos_sim[best_idx].item()
        best_class = class_names[best_idx]

        class_scores = {
            class_names[i]: cos_sim[i].item()
            for i in range(len(class_names))
        }

        return best_class, best_score, class_scores

    def inference(self, img_bgr, embedding_matrix, img_id=1, categories=None, img_yolo_annotation=None):
        annotations = []
        annotations_to_check = []
        if not img_yolo_annotation:
            img_yolo_annotation = self.yd.inference(img_bgr=img_bgr, img_id=img_id)

        annotations += img_yolo_annotation

        for ann in img_yolo_annotation:
            if ann['category_name'] != 'Gripper':
                x, y, w, h = ann['bbox']
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                img_bgr[y1:y2, x1:x2] = 0

        # plt.figure(figsize=(5, 5))
        # plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        # plt.title(f"img_bgr")
        # plt.show()

        masks = self.sa.inference(img_bgr)
        #self.sa.visualization_polygon(img_bgr, masks)
        crops, coco_bboxes, coco_areas = self.sa.get_masked_crops(img_bgr, masks)


        for crop_bgr, coco_bbox, coco_area in zip(crops, coco_bboxes, coco_areas):
            best_class, best_score, class_scores = self.compare(crop_bgr, embedding_matrix)
            #print(f'class_scores {class_scores}')

            if best_score > 0.7:
                if categories:
                    category_id = [item['id'] for item in categories if item['name'] == best_class][0]
                else:
                    category_id = None

                ann = {
                    "id": str(uuid.uuid4()),
                    "image_id": img_id,
                    "category_id": category_id,
                    "category_name": best_class,
                    "bbox": coco_bbox,
                    "area": coco_area,
                    "iscrowd": 0,
                    "confidence": float(best_score),
                    "comment": 'few_shot_detection'
                }
                if best_score > 0.85:
                    annotations.append(ann)
                else:
                    annotations_to_check.append(ann)

            # plt.figure(figsize=(5, 5))
            # plt.imshow(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            # plt.axis("off")
            # plt.title(f"best_class {best_class}, best_score {best_score}")
            # plt.show()

        return annotations, annotations_to_check

    def coco_yolo_exists(self, coco_yolo_json, embedding_matrix):
        print(f'Подгружен файл с coco аннотацией {coco_yolo_json}')
        with open(Path(coco_yolo_json), 'r', encoding='utf-8') as f:
            coco_yolo = json.load(f)
            yolo_categories = coco_yolo['categories']

        additional_categories = [{"id": len(yolo_categories)+i, "name": cat_name} for i, cat_name in enumerate(embedding_matrix.keys())]
        total_categories = yolo_categories + additional_categories
        coco = coco_yolo.copy()
        coco['categories'] = total_categories

        coco_to_check = coco.copy()
        coco_to_check['annotations'] = []

        images = coco['images']
        for image in tqdm(images):
            img_path = image['file_name']
            img_id = image['id']
            img_yolo_annotation = [item for item in coco['annotations'] if item['image_id'] == img_id]
            img_bgr = cv2.imread(img_path)


            upd_img_annotations, annotations_to_check = self.inference(img_bgr=img_bgr,
                                       embedding_matrix=embedding_matrix,
                                       img_id=img_id,
                                       categories=total_categories,
                                       img_yolo_annotation=img_yolo_annotation)

            coco["annotations"] = [item for item in coco["annotations"] if item['image_id'] != img_id] + upd_img_annotations
            coco_to_check['annotations'] += annotations_to_check
        return coco, coco_to_check


    def coco_yolo_not_exist(self, images_folder_path, yolo_categories, embedding_matrix):
        additional_categories = [{"id": len(yolo_categories) + i, "name": cat_name} for i, cat_name in
                                 enumerate(embedding_matrix.keys())]
        total_categories = yolo_categories + additional_categories
        coco = {"images": [], "annotations": [], "categories": total_categories}

        coco_to_check = coco.copy()
        coco_to_check['annotations'] = []

        img_id = 0
        for img_path in tqdm(sorted(Path(images_folder_path).glob("*/*.jpg"))):
            img_id += 1
            img_bgr = cv2.imread(str(img_path))
            h, w = img_bgr.shape[:2]

            upd_img_annotations, annotations_to_check = self.inference(img_bgr=img_bgr,
                                               embedding_matrix=embedding_matrix,
                                               img_id=img_id,
                                               categories=total_categories)


            coco["images"].append({"id": img_id, "file_name": str(img_path), "width": w, "height": h})
            coco["annotations"] += upd_img_annotations
            coco_to_check['annotations'] += annotations_to_check

        return coco, coco_to_check

def main(images_folder_path='data/test',
         coco_annotation_json='data/test/final_coco_annotation.json',
         coco_yolo_annotation_json='data/test/coco_yolo_annotation.json',
         coco_to_check_annotation_json='data/test/coco_to_check_annotation.json'):

    sl = SelfLabeling(emb_backbone='dinov2')
    embedding_matrix = sl.create_embedding_matrix(data_dir=Path('data/new_classes'))

    if Path(coco_yolo_annotation_json).is_file():
        coco, coco_to_check = sl.coco_yolo_exists(coco_yolo_annotation_json, embedding_matrix)

    else:
        yolo_categories = sl.yd.get_yolo_categories()
        coco, coco_to_check = sl.coco_yolo_not_exist(images_folder_path, yolo_categories, embedding_matrix)

    with open(coco_annotation_json, "w") as f:
        json.dump(coco, f, indent=2)

    with open(coco_to_check_annotation_json, "w") as f:
        json.dump(coco_to_check, f, indent=2)


if __name__ == "__main__":
    main()
