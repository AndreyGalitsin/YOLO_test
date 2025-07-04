import json, cv2, torch
from ultralytics import YOLO
from pathlib import Path
import uuid


class YOLODetection:
    def __init__(self, conf_th=0.9):
        self.conf_th = conf_th
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("checkpoints/yolo_marking.pt")
        self.model.fuse()
        self.model.to(self.device).eval()

    def get_yolo_categories(self):
        return [{"id": key, "name": self.model.names[key]} for key in self.model.names]

    def inference(self, img_bgr, img_id=1):
        res = self.model.predict(
            img_bgr,
            conf=self.conf_th,
            device=self.device,
            verbose=False
        )[0]

        annotations = []
        for box, cls, score in zip(
                res.boxes.xyxy.cpu(),
                res.boxes.cls.cpu(),
                res.boxes.conf.cpu()
        ):
            x1, y1, x2, y2 = box.tolist()
            annotations.append({
                "id": str(uuid.uuid4()),
                "image_id": img_id,
                "category_id": int(cls),
                "category_name": self.model.names[int(cls)],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
                "confidence": float(score),
                "comment": 'yolo_detection'
            })

        return annotations

def demo(images_folder_path, out_json="data/coco_yolo.json"):
    yd = YOLODetection()
    categories = yd.get_yolo_categories()
    coco = {"images": [], "annotations": [], "categories": categories}
    for img_id, img_path in enumerate(sorted(Path(images_folder_path).glob("*/*.jpg"))):
        img_bgr = cv2.imread(str(img_path))
        annotations = yd.inference(img_bgr=img_bgr, img_id=img_id)
        h, w = img_bgr.shape[:2]
        coco["images"].append(
            {"id": img_id, "file_name": str(img_path), "width": w, "height": h}
        )
        coco["annotations"] += annotations

    with open(out_json, "w") as f:
        json.dump(coco, f, indent=2)


if __name__ == "__main__":
    demo(images_folder_path='data/task_169')
