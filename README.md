# HelmetGuard AI: Real-Time Motorcycle Helmet Detection System

HelmetGuard AI is an automated computer-vision system that detects whether motorcycle riders are wearing helmets from uploaded images/video or a live webcam/camera feed. 

It combines a fine-tuned **YOLO object-detection model**, a **FastAPI backend (Python)**, and a modern **Next.js (React) Dashboard**.

## Features

- **Automated Detection:** Detects "person", "helmet", and "no-helmet" classes with bounding boxes and confidence scores.
- **Multiple Inputs:** Supports both static image/video uploads and real-time webcam inference.
- **Live Dashboard:** Clean Next.js dashboard showing real-time detections, confidence adjustments, and aggregate compliance stats (compliant vs. non-compliant riders).
- **Dockerized Setup:** Easily run the entire stack (Frontend + Backend API) using a single Docker Compose command.
- **YOLOv11 Model:** Built using the latest state-of-the-art YOLOv11 architecture (small variant) for an optimal balance of speed and accuracy.

---

## Tech Stack

- **Frontend:** [Next.js](https://nextjs.org/) (React), Tailwind CSS, Lucide Icons, Recharts.
- **Backend / API:** [FastAPI](https://fastapi.tiangolo.com/), Uvicorn, Python.
- **Machine Learning:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), OpenCV, PyTorch.
- **Infrastructure:** Docker, Docker Compose.

---

## Project Structure

- `/web-client` - The Next.js frontend dashboard.
- `/ml-service` - The FastAPI backend and YOLO training/inference scripts.
- `/unified_dataset` - The dataset structure used for training the model.
- `docker-compose.yml` - Orchestration file to run both services simultaneously.

---

## Getting Started (Local Development)

You can run this project easily using Docker Compose or manually.

### Option 1: Using Docker Compose (Recommended)
This will start both the Next.js frontend and the FastAPI backend. Ensure you have Docker installed and running.

```bash
# From the root directory, run:
docker-compose up --build
```

- The Web Dashboard will be available at: `http://localhost:3000`
- The ML API Service will be available at: `http://localhost:8000`

### Option 2: Running Services Manually

**1. Start the ML Service API**
```bash
cd ml-service
python -m venv venv
# Windows: .\venv\Scripts\activate | Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**2. Start the Web Client**
Open a new terminal window:
```bash
cd web-client
npm install
npm run dev
```

---

## Model Training

If you'd like to retrain or fine-tune the model on your own data:

1. Prepare your YOLO formatted dataset in the `/unified_dataset` folder (update `data.yaml` accordingly). 
   *Note: The current model was trained on a robust dataset of **8,328 images**.*
2. Run the training script (configured for **50 epochs**):
   ```bash
   cd ml-service
   python training/train.py
   ```
3. The script is configured to automatically resume from `last.pt` if training was interrupted. It outputs the trained models to `ml-service/runs/...`.

---

## Tuning Detection Confidence

Inside the Web Dashboard, you can adjust the **Confidence Threshold**. 
- Lower it (e.g., `0.3`) if the model is missing helmets (increases recall, but risks false positives).
- Raise it (e.g., `0.6`) if the model is drawing boxes around incorrect objects (increases precision).

## License

This project is open-source. Feel free to fork, modify, and use it for your own research or portfolio!
