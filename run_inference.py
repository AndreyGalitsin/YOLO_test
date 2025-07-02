# -*- coding: utf-8 -*-
"""
----------------------
Pipeline step 1:  «Generation COCO JSON with YOLO predictions»

Input
-----
1. --images_folder_path       : folder with *.jpg images (one level, any name)

Output
------
* yolo_pred.json        – COCO JSON with YOLO predictions
"""

import json, cv2, torch
from ultralytics import YOLO
from pathlib import Path

def main(images_folder_path, out_json="data/task_169/yolo_pred.json", conf_th=0.25):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO("checkpoints/yolo_marking.pt")
    model.fuse()
    model.to(device).eval()

    categories = [{"id": key, "name": model.names[key]} for key in model.names]
    coco = {"images": [], "annotations": [], "categories": categories}
    ann_id = 1
    img_id = 0
    for img_id, img_path in enumerate(sorted(Path(images_folder_path).glob("*/*.jpg"))):
        im = cv2.imread(str(img_path))
        h, w = im.shape[:2]

        res = model.predict(
            im,
            conf=conf_th,
            device=device,
            verbose=False
        )[0]

        coco["images"].append(
            {"id": img_id, "file_name": str(img_path), "width": w, "height": h}
        )

        for box, cls, score in zip(
                res.boxes.xyxy.cpu(),
                res.boxes.cls.cpu(),
                res.boxes.conf.cpu()
        ):
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
            ann_id += 1


    with open(out_json, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"Saved {ann_id-1} detections → {out_json}, nof_images {img_id+1}")

if __name__ == "__main__":
    main(images_folder_path='data/task_169')
