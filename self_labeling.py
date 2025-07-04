from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import open_clip
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle

from sample_anything import SampleAnything
import torch.nn.functional as F
from transformers import AutoImageProcessor, Dinov2Model
from torchvision import transforms

from yolo_detection import YOLODetection


class SelfLabeling:
    def __init__(self, emb_backbone="clip"):
        self.backbone = emb_backbone
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sa = SampleAnything()
        self.yd = YOLODetection()

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

    def create_embedding_matrix(self, data_dir: Path, pkl_path: Path=Path('data/embedding_matrix.pkl')) -> dict[str, torch.Tensor]:
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
            crops = self.sa.get_masked_crops(img_bgr, [max_area_mask])
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

    def inference(self, img_bgr, embedding_matrix, img_id=1):
        annotations = self.yd.inference(img_bgr=img_bgr, img_id=img_id)
        for ann in annotations:
            if ann['category_name'] != 'Gripper':
                x, y, w, h = ann['bbox']
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                img_bgr[y1:y2, x1:x2] = 0

        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"img_bgr")
        plt.show()

        masks = self.sa.inference(img_bgr)
        self.sa.visualization_polygon(img_bgr, masks)
        crops = self.sa.get_masked_crops(img_bgr, masks)
        for crop_bgr in crops:
            best_class, best_score, class_scores = self.compare(crop_bgr, embedding_matrix)
            print(f'class_scores {class_scores}')
            plt.figure(figsize=(5, 5))
            plt.imshow(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"best_class {best_class}, best_score {best_score}")
            plt.show()

    def main(self, images_folder_path):
        yolo_categories = [{"id": key, "name": self.yolo_model.names[key]} for key in self.yolo_model.names]
        next_category_id = max(self.yolo_model.names.keys()) + 1
        additional_categories = [{"id": next_category_id+i, "name": cat_name} for i, cat_name in enumerate(self.reference_embeddings.keys())]


def demo():
    backbones = ["clip", 'dinov2']

    sl = SelfLabeling(emb_backbone=backbones[1])
    embedding_matrix = sl.create_embedding_matrix(data_dir=Path('data/new_classes'))
    img_path = 'data/task_169/variation_613/task_169--variation_613--episode_113401--camera_1--middle--samokat_grapefruit_juice.jpg'
    # img_path = "/home/andrey/Andrey_projects/YOLO_test/data/task_169/variation_622/task_169--variation_622--episode_103470--camera_2--middle--lay's_crab.jpg"
    # img_path = '/home/andrey/Andrey_projects/YOLO_test/data/task_169/variation_629/task_169--variation_629--episode_95667--camera_2--middle--samokat_buckwheat_flakes.jpg'
    img_path = '/home/andrey/Andrey_projects/YOLO_test/data/task_169/variation_624/task_169--variation_624--episode_87482--camera_3--middle--samokat_sunflower_seed_oil.jpg'
    img_bgr = cv2.imread(img_path)
    sl.inference(img_bgr, embedding_matrix)


if __name__ == "__main__":
    demo()
