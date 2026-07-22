"""
Evaluate the served model on the HELD-OUT TEST split (never seen in training or
tuning). Produces per-class precision/recall/mAP50/mAP50-95, a confusion matrix,
and PR/F1 curves, and prints the no_helmet row explicitly.

Run AFTER training completes:
    python training/validate.py

The `iou` and `conf` here mirror what the FastAPI backend serves so validated
behavior == served behavior (no silent mismatch).
"""

import os
from ultralytics import YOLO

# Must match app/inference.py inference settings.
SERVED_IOU = 0.7
SERVED_CONF = 0.25          # keep in sync with app.inference.DEFAULT_CONF
RUN_NAME = "helmet_v4_clean"


def find_weights():
    best = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runs", RUN_NAME, "weights", "best.pt"))
    if not os.path.isfile(best):
        raise FileNotFoundError(f"Trained weights not found at {best}. Run training/train.py first.")
    return best


def main():
    data_yaml = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "unified_dataset", "data.yaml")
    )
    weights = find_weights()
    print(f"Evaluating {weights}\n on TEST split of {data_yaml}\n")

    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml,
        split="test",           # the untouched held-out set
        iou=SERVED_IOU,
        conf=SERVED_CONF,
        plots=True,             # confusion_matrix.png, PR/F1 curves
        save_json=True,
        project=os.path.abspath("runs"),
        name=f"{RUN_NAME}_test_eval",
        exist_ok=True,
    )

    names = model.names
    print("\n================ PER-CLASS METRICS (TEST) ================")
    print(f"{'class':<12}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
    for i, c in enumerate(metrics.box.ap_class_index):
        p = metrics.box.p[i]
        r = metrics.box.r[i]
        ap50 = metrics.box.ap50[i]
        ap = metrics.box.ap[i]
        print(f"{names[c]:<12}{p:>8.3f}{r:>8.3f}{ap50:>8.3f}{ap:>10.3f}")
    print("----------------------------------------------------------")
    print(f"{'all (mean)':<12}{metrics.box.mp:>8.3f}{metrics.box.mr:>8.3f}"
          f"{metrics.box.map50:>8.3f}{metrics.box.map:>10.3f}")
    print("==========================================================")
    print(f"\nArtifacts saved under runs/{RUN_NAME}_test_eval/")


if __name__ == "__main__":
    main()
