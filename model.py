# model.py
from ultralytics import YOLO
import cv2
import os

# โหลดโมเดลเพียงครั้งเดียว
model = YOLO(r"C:\Users\User\Desktop\ai-model-api\best.pt")

def model_inference(filename: str):
    file_path = os.path.join("models", filename)

    if not os.path.exists(file_path):
        return {"error": "Image not found."}

    # โหลดภาพ
    img = cv2.imread(file_path)

    # ตรวจจับ
    results = model.predict(source=img, conf=0.5, iou=0.5, verbose=False)
    boxes = results[0].boxes

    response = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        response.append({
            "label": label,
            "confidence": round(confidence, 4),
            "box": [round(x1), round(y1), round(x2), round(y2)]
        })

    return {"detections": response}
