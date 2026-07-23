import time
import uuid
import threading
import numpy as np
from ultralytics import YOLO
from typing import Dict, Any
import os

# Inference settings -- kept in sync with training/validate.py so the model's
# SERVED behavior matches its VALIDATED behavior (no silent mismatch).
DEFAULT_CONF = 0.25
SERVED_IOU = 0.45
AGNOSTIC_NMS = True     # suppress overlapping boxes of different classes

_RUNS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs"))

# The ONLY model the backend will serve: the validated, leak-free run.
SERVED_RUN = "helmet_v4_clean"
SERVED_WEIGHTS = os.path.join(_RUNS, SERVED_RUN, "weights", "best.pt")

# Opt-in escape hatch for local experimentation ONLY. When set truthy, a missing
# validated model falls back to COCO-pretrained yolo11n.pt instead of refusing to
# start. Never set this in production -- it would serve an unvalidated model.
_ALLOW_UNVALIDATED = os.getenv("ALLOW_UNVALIDATED_MODEL", "").lower() in ("1", "true", "yes")

_model = None
_model_lock = threading.Lock()


class ModelNotAvailableError(RuntimeError):
    """Raised when the validated model is missing and fallback is not allowed."""


def _resolve_weights() -> str:
    """Return the path to the validated model, or fail loudly.

    We deliberately do NOT fall back to older/archived runs or to a raw COCO
    checkpoint by default. Serving a deprecated or unvalidated model silently is
    exactly the failure mode we want to prevent: the app should refuse to start
    rather than pretend an untrained/leaky model is the real detector.
    """
    if os.path.isfile(SERVED_WEIGHTS):
        return SERVED_WEIGHTS

    if _ALLOW_UNVALIDATED:
        print(
            "WARNING: validated model not found; ALLOW_UNVALIDATED_MODEL is set, "
            "falling back to pretrained yolo11n.pt. DO NOT use this in production."
        )
        return "yolo11n.pt"

    raise ModelNotAvailableError(
        f"Validated model not found at {SERVED_WEIGHTS}.\n"
        f"Train it first:  python training/train.py  (then: python training/validate.py)\n"
        f"The backend refuses to serve a deprecated/unvalidated model. For local "
        f"experimentation only, set ALLOW_UNVALIDATED_MODEL=1 to fall back to a "
        f"pretrained COCO checkpoint (NOT for production)."
    )


def load_model():
    """Load the model exactly once (thread-safe). Safe to call at startup and
    from concurrent requests -- the double-checked lock guarantees a single load.
    Raises ModelNotAvailableError if no validated model exists and fallback is
    not explicitly allowed."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:                    # re-check inside the lock
                weights = _resolve_weights()
                print(f"Loading detection model: {weights}")
                _model = YOLO(weights)
                # FORCE PyTorch initialization and layer fusion SYNCHRONOUSLY 
                # before allowing concurrent web requests. Otherwise, concurrent 
                # webcam frames can race during the first predict() and cause:
                # AttributeError: 'Conv' object has no attribute 'bn'
                print("Warming up model with dummy inference to fuse layers...")
                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                _model.predict(source=dummy_img, verbose=False)
                print("Model warmed up successfully.")
    return _model


def get_model():
    return load_model()


def run_inference(image: np.ndarray, confidence_threshold: float = DEFAULT_CONF) -> Dict[str, Any]:
    start_time = time.time()
    yolo_model = get_model()

    results = yolo_model.predict(
        source=image,
        conf=confidence_threshold,
        iou=SERVED_IOU,
        agnostic_nms=AGNOSTIC_NMS,
        save=False,
        verbose=False,
    )
    inference_time_ms = int((time.time() - start_time) * 1000)
    
    detections = []
    compliant = 0      # helmet detections
    violations = 0     # no_helmet detections (a bare head IS a violation -- Strategy A)

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

            if cls_name == "helmet":
                compliant += 1
            elif cls_name == "no_helmet":
                violations += 1

    # This is a 2-class model (helmet / no_helmet). Every detected head is a
    # rider we assessed, so total_riders = compliant heads + violation heads.
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
