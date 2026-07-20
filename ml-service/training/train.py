import os
from ultralytics import YOLO

def main():
    # Check if a previous model exists to fine-tune
    best_weights_path = os.path.abspath(os.path.join("runs", "helmet_v2_accurate-2", "weights", "best.pt"))
    
    if os.path.exists(best_weights_path):
        print(f"Found existing best model at {best_weights_path}")
        print("Fine-tuning on the updated dataset with hard negatives...")
        model = YOLO(best_weights_path)
        resume_training = False # Start a new fine-tuning run
    else:
        print("Loading YOLO11s pretrained model (The absolute latest YOLO version!)...")
        model = YOLO("yolo11s.pt") 
        resume_training = False
    
    # The absolute path to the massive combined dataset
    data_yaml_path = r"d:\University Work\Semester 8\Helmet Detection System\unified_dataset\data.yaml"
    
    print(f"Starting training on massive unified dataset at: {data_yaml_path}")
    
    # Run training
    # - epochs=100 for a deep, high-accuracy training run
    # - imgsz=640 is standard YOLO input size
    # - patience=20 stops training early if the model stops improving
    
    train_args = {
        "data": data_yaml_path,
        "epochs": 10, # Just 10 epochs for fine-tuning
        "patience": 5, 
        "imgsz": 640,
        "batch": 16,
        "project": os.path.abspath("runs"),
        "name": "helmet_v3_hard_negatives",
        "device": "cpu",
        "exist_ok": True
    }
    
    if resume_training:
        results = model.train(resume=True)
    else:
        results = model.train(**train_args)
    
    print("Training finished!")
    print("Your trained model weights are saved at: runs/helmet_v3_hard_negatives/weights/best.pt")

if __name__ == "__main__":
    main()
