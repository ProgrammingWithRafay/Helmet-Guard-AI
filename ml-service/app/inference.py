import time
import uuid
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Any, List
import os

model = None

def get_model():
    global model
    if model is None:
        v3_best = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs", "helmet_v3_hard_negatives", "weights", "best.pt"))
        v2_best = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs", "helmet_v2_accurate-2", "weights", "best.pt"))
        v2_last = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs", "helmet_v2_accurate-2", "weights", "last.pt"))
        v1_best = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs", "detect", "runs", "helmet_v1", "weights", "best.pt"))
        
        if os.path.exists(v3_best):
            weights_path = v3_best
            print(f"Using fine-tuned v3 model (Hard Negatives): {weights_path}")
        elif os.path.exists(v2_best):
            weights_path = v2_best
            print(f"Using completed v2 model: {weights_path}")
        elif os.path.exists(v2_last):
            weights_path = v2_last
            print(f"Testing live training checkpoint: {weights_path}")
        elif os.path.exists(v1_best):
            weights_path = v1_best
            print(f"Using stable v1 model: {weights_path}")
        else:
            # Fallback to yolo11n if no trained models are found
            print("Warning: No trained models found. Using pretrained yolo11n.pt")
            weights_path = "yolo11n.pt"
            
        model = YOLO(weights_path)
    return model

def run_inference(image: np.ndarray, confidence_threshold: float = 0.5) -> Dict[str, Any]:
    start_time = time.time()
    yolo_model = get_model()
    
    # Run prediction with agnostic_nms=True to suppress overlapping boxes of different classes
    results = yolo_model.predict(source=image, conf=confidence_threshold, save=False, agnostic_nms=True)
    
    inference_time_ms = int((time.time() - start_time) * 1000)
    
    detections = []
    total_riders = 0
    compliant = 0
    violations = 0
    
    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = yolo_model.names[cls_id]
            
            det_id = f"det_{uuid.uuid4().hex[:8]}"
            detections.append({
                "id": det_id,
                "class": cls_name,
                "confidence": conf,
                "bbox": {
                    "x_min": int(x1),
                    "y_min": int(y1),
                    "x_max": int(x2),
                    "y_max": int(y2)
                }
            })
            
            if cls_name == "person" or cls_name == "rider":
                total_riders += 1
            elif cls_name == "helmet":
                compliant += 1
            elif cls_name == "no_helmet":
                violations += 1

    # Heuristic for datasets that only label 'helmet' and 'no_helmet'
    if total_riders == 0:
        total_riders = compliant + violations
        
    return {
        "request_id": str(uuid.uuid4())[:8],
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "inference_time_ms": inference_time_ms,
        "detections": detections,
        "summary": {
            "total_riders": total_riders,
            "compliant": compliant,
            "violations": violations
        }
    }
