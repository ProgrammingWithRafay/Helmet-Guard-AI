import cv2
import urllib.request
import os
import uuid
import numpy as np

# A list of image URLs containing people without helmets (e.g., standard faces)
image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/a/a0/Pierre-Person.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/e/e0/Placeholder_person.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/a/a8/Bill_Gates_2017_%28cropped%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/6/68/Joe_Biden_presidential_portrait.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/b/b8/LeBron_James_%2851959977144%29_%28cropped2%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/d/d4/Keanu_Reeves_2019.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/c/c1/Emma_Watson_2013.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/1/1a/Tom_Cruise_by_Gage_Skidmore_2.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/f/f3/Nicolas_Cage_-_66%C3%A8me_Festival_de_Venise_%28Mostra%29.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/9/98/Morgan_Freeman_at_the_2018_Make-A-Wish_Gala.jpg"
]

# We need Haar Cascade for Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Paths for dataset
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "unified_dataset"))
train_images_dir = os.path.join(dataset_path, "images", "train")
train_labels_dir = os.path.join(dataset_path, "labels", "train")
val_images_dir = os.path.join(dataset_path, "images", "valid")
val_labels_dir = os.path.join(dataset_path, "labels", "valid")

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

print(f"Adding hard negatives to {train_images_dir} and {val_images_dir}")

for i, url in enumerate(image_urls):
    try:
        # Request with a standard User-Agent so we don't get blocked
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        
        if img is None:
            print(f"Failed to decode {url}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Split into train/val
        target_img_dir = train_images_dir if i % 5 != 0 else val_images_dir
        target_lbl_dir = train_labels_dir if i % 5 != 0 else val_labels_dir
        
        if len(faces) == 0:
            # If no face is detected by OpenCV, we can just save it as a background image (empty label file)
            filename = f"bg_{uuid.uuid4().hex[:8]}"
            cv2.imwrite(os.path.join(target_img_dir, f"{filename}.jpg"), img)
            open(os.path.join(target_lbl_dir, f"{filename}.txt"), 'w').close()
            print(f"Added background: {filename}")
        else:
            filename = f"face_{uuid.uuid4().hex[:8]}"
            # To ensure the model doesn't overfit on these exact faces, save a few copies with data augmentation (flips)
            cv2.imwrite(os.path.join(target_img_dir, f"{filename}.jpg"), img)
            img_flipped = cv2.flip(img, 1)
            cv2.imwrite(os.path.join(target_img_dir, f"{filename}_flip.jpg"), img_flipped)
            
            # Create YOLO format labels
            h, w, _ = img.shape
            
            # Write normal labels
            with open(os.path.join(target_lbl_dir, f"{filename}.txt"), 'w') as f, open(os.path.join(target_lbl_dir, f"{filename}_flip.txt"), 'w') as f_flip:
                for (x, y, fw, fh) in faces:
                    # Expand the face box slightly to include the whole head
                    head_x = max(0, x - int(fw * 0.2))
                    head_y = max(0, y - int(fh * 0.4))
                    head_w = min(w - head_x, int(fw * 1.4))
                    head_h = min(h - head_y, int(fh * 1.6))
                    
                    # YOLO format: class x_center y_center width height (normalized)
                    x_center = (head_x + head_w / 2.0) / w
                    y_center = (head_y + head_h / 2.0) / h
                    norm_w = head_w / w
                    norm_h = head_h / h
                    
                    # class 1 is no_helmet
                    f.write(f"1 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                    
                    # Flipped labels (x coordinate is mirrored)
                    f_flip.write(f"1 {1.0 - x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                    
            print(f"Added face (no_helmet): {filename} with {len(faces)} faces")
            
    except Exception as e:
        print(f"Error processing {url}: {e}")

# We will also add some synthetic background images (noise, solid colors, gradients) to teach it not to hallucinate shapes
for i in range(20):
    bg_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    filename = f"bg_noise_{uuid.uuid4().hex[:8]}"
    target_img_dir = train_images_dir if i % 5 != 0 else val_images_dir
    target_lbl_dir = train_labels_dir if i % 5 != 0 else val_labels_dir
    cv2.imwrite(os.path.join(target_img_dir, f"{filename}.jpg"), bg_img)
    open(os.path.join(target_lbl_dir, f"{filename}.txt"), 'w').close()

for i in range(20):
    bg_img = np.zeros((640, 640, 3), dtype=np.uint8)
    bg_img[:] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    filename = f"bg_solid_{uuid.uuid4().hex[:8]}"
    target_img_dir = train_images_dir if i % 5 != 0 else val_images_dir
    target_lbl_dir = train_labels_dir if i % 5 != 0 else val_labels_dir
    cv2.imwrite(os.path.join(target_img_dir, f"{filename}.jpg"), bg_img)
    open(os.path.join(target_lbl_dir, f"{filename}.txt"), 'w').close()

print("Hard negatives added successfully. Ready for retraining.")
