# HelmetGuard AI: Real-Time Motorcycle Helmet Detection System

HelmetGuard AI is an automated computer-vision system that detects whether motorcycle riders are wearing helmets from uploaded images/video or a live webcam/camera feed. 

It combines a fine-tuned **YOLO object-detection model**, a **FastAPI backend (Python)**, and a modern **Next.js (React) Dashboard**.

## Features

- **Automated Detection:** Detects two classes — `helmet` and `no_helmet` — with bounding boxes and confidence scores. A `no_helmet` detection means a **bare head**, which the system treats as a **compliance violation** (see "Class semantics" below).
- **Multiple Inputs:** Supports both static image/video uploads and real-time webcam inference.
- **Live Dashboard:** Clean Next.js dashboard showing real-time detections, confidence adjustments, and aggregate compliance stats (compliant vs. non-compliant riders).
- **Dockerized Setup:** Easily run the entire stack (Frontend + Backend API) using a single Docker Compose command.
- **YOLOv11 Model:** Built using the latest state-of-the-art YOLOv11 architecture (small variant) for an optimal balance of speed and accuracy.

---

## Tech Stack

- **Frontend:** [Next.js](https://nextjs.org/) (React), Tailwind CSS, Lucide Icons, Recharts.
- **UI/UX Design:** [Figma](https://www.figma.com/) (Prototyping & Wireframing).
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

## Pre-Trained Model (Hugging Face)

To avoid bloating the Git repository with large binary files, the final trained YOLO weights (`best.pt`) are hosted on Hugging Face.

You can view and download the production-ready model here:
**[Hugging Face: ProgrammingWithRafay/helmet-detection-yolo](https://huggingface.co/ProgrammingWithRafay/helmet-detection-yolo)**

Place the downloaded `best.pt` file inside the `ml-service/runs/helmet_v4_clean/weights/` folder (or wherever your environment variables dictate) for the FastAPI backend to use it.

---

## Model Training

If you'd like to retrain or fine-tune the model on your own data:

1. Prepare your YOLO formatted dataset in the `/unified_dataset` folder (update `data.yaml` accordingly). 
   *Note: The current model was trained on a robust dataset of **8,358 images** (8,328 street images + 30 custom presentation environment images). Due to computational requirements, the final production model was fine-tuned using **Kaggle Cloud GPUs** for significantly accelerated processing to combat Domain Shift for live presentations.*
2. Run the local training script (starts fresh from COCO-pretrained `yolo11s.pt`; caps at 60 epochs with early stopping via `patience=10`):
   ```bash
   cd ml-service
   python training/train.py
   ```
3. The script is configured to automatically resume from `last.pt` if training was interrupted. It outputs the trained models to `ml-service/runs/...`.

---

## Class Semantics (important)

The model has exactly **two classes**: `helmet` (index 0) and `no_helmet` (index 1). There is **no** separate `person`/`rider` class.

**`no_helmet` means "a bare human head."** A bare head is a **positive, detectable violation** — not a background/negative to be ignored. The dashboard counts each `no_helmet` detection as a compliance violation and each `helmet` detection as compliant.

**Consequence to be aware of:** because the model flags *any* bare head, it will report a violation for a bare-headed pedestrian in frame, not only a motorcycle rider. Distinguishing "rider without helmet" from "any bare head" would require adding a `rider`/`person` class and re-labeling the dataset (deliberately out of scope for the current model).

> Note: the standalone `training/add_hard_negatives.py` utility fetches face images and labels bare heads as `no_helmet` (positive), consistent with the above. It is **not** wired into the training pipeline and its output is not part of the current dataset.

---

## Tuning Detection Confidence

Inside the Web Dashboard, you can adjust the **Confidence Threshold**. 
- Lower it (e.g., `0.3`) if the model is missing detections (increases recall — catches more helmets *and* more bare-head violations — but risks false positives).
- Raise it (e.g., `0.6`) if the model is drawing boxes around incorrect objects (increases precision).

## License

This project is open-source. Feel free to fork, modify, and use it for your own research or portfolio!
