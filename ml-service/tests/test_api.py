import io
import base64
import cv2
import numpy as np
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def create_dummy_image():
    # Create a simple blank image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded = cv2.imencode('.jpg', img)
    return encoded.tobytes()

def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_detect_image_valid():
    img_bytes = create_dummy_image()
    response = client.post(
        "/api/v1/detect/image",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"confidence_threshold": 0.5}
    )
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "detections" in data
    assert "summary" in data

def test_detect_image_invalid_file():
    response = client.post(
        "/api/v1/detect/image",
        files={"file": ("test.txt", b"hello world", "text/plain")},
        data={"confidence_threshold": 0.5}
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "INVALID_FILE_TYPE"

def test_detect_frame_valid():
    img_bytes = create_dummy_image()
    b64_img = base64.b64encode(img_bytes).decode('utf-8')
    payload = {
        "frame_base64": f"data:image/jpeg;base64,{b64_img}",
        "confidence_threshold": 0.5
    }
    response = client.post("/api/v1/detect/frame", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data

def test_detect_frame_invalid():
    payload = {
        "frame_base64": "invalid_base64_data_here",
        "confidence_threshold": 0.5
    }
    response = client.post("/api/v1/detect/frame", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "NO_FRAME_DATA"
