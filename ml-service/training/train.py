import os
import logging
import torch
from ultralytics import YOLO

# Fresh, leak-free retrain ("v4_clean").
#
# CRITICAL: we start from COCO-pretrained yolo11s.pt, NOT from any earlier
# helmet_v2/v3 weights. Those older weights were trained on the pre-fix dataset,
# whose frames now live in our leak-free valid/test splits -- fine-tuning from
# them would reintroduce leakage at the weight level and inflate metrics. A clean
# COCO backbone has never seen this data, so evaluation stays honest.
#
# AUTO-RESUME: if a previous run of THIS script was interrupted (session timeout,
# machine sleep/restart), it resumes from runs/helmet_v4_clean/weights/last.pt --
# i.e. only its OWN clean checkpoint, never the archived leaky v2/v3 weights.

RUN_NAME = "helmet_v4_clean"


def _setup_file_logging(runs_dir: str) -> str:
    """Tee Ultralytics' epoch summaries to a log file so progress is checkable
    without watching the terminal (the per-batch progress bar stays on stdout;
    per-epoch summary lines and messages are captured here)."""
    log_path = os.path.join(runs_dir, f"{RUN_NAME}_train.log")
    os.makedirs(runs_dir, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger("ultralytics").addHandler(fh)
    return log_path


def main():
    data_yaml_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "unified_dataset", "data.yaml")
    )
    runs_dir = os.path.abspath("runs")
    last_ckpt = os.path.join(runs_dir, RUN_NAME, "weights", "last.pt")

    log_path = _setup_file_logging(runs_dir)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}  ({'GPU' if device == 0 else 'CPU'})")
    print(f"Dataset: {data_yaml_path}")
    print(f"Progress log: {log_path}")

    if os.path.isfile(last_ckpt):
        # Resume our own interrupted clean run. Ultralytics reloads all prior
        # train args from the checkpoint, so we pass only resume=True.
        print(f"Found interrupted checkpoint -> resuming from {last_ckpt}")
        model = YOLO(last_ckpt)
        model.train(resume=True)
    else:
        print("Starting fresh from COCO-pretrained yolo11s.pt (no leaky-weight fine-tune).")
        model = YOLO("yolo11s.pt")
        # Prior runs plateaued on val/cls_loss around epoch ~14, so a high epoch
        # cap with early stopping (patience) avoids wasting compute past plateau.
        model.train(
            data=data_yaml_path,
            epochs=60,          # cap; early stopping usually ends sooner
            patience=10,        # stop if val metric doesn't improve for 10 epochs
            imgsz=640,
            batch=16,           # safe on T4/P100 (16GB); raise to 32/64 on bigger GPUs
            project=runs_dir,
            name=RUN_NAME,
            device=device,
            workers=8,
            seed=0,
            deterministic=True,
            exist_ok=True,
            plots=True,         # PR/F1 curves, confusion matrix, results.png
            # Fixed-camera use case: conservative geometry (no vertical flip / rotation).
            degrees=0.0,
            flipud=0.0,
            fliplr=0.5,
        )

    print("Training finished.")
    print(f"Best weights: runs/{RUN_NAME}/weights/best.pt")
    print("Next: run `python training/validate.py` to evaluate on the held-out test split.")


if __name__ == "__main__":
    main()
