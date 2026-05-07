import os
import glob
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import Sam3Model, Sam3Processor
import time
from datetime import timedelta

# --- magic bug fix by gemini ---
import huggingface_hub.dataclasses
huggingface_hub.dataclasses.type_validator = lambda *args, **kwargs: None
# ---- thank you mr clanker -----

# Import evaluation functions
import sys
sys.path.append(r"C:\Users\x3non\OneDrive\Desktop\A03-AE2224")  
from scoring_general import CATEGORY_COLOURS, build_colour_index, evaluate_pair, print_results

# ==========================================
# FWIoU & Confusion Matrix Helper Functions
# ==========================================
def fast_hist(a, b, n):
    """Calculates a fast 2D confusion matrix (histogram)."""
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def calculate_fwiou(confusion_matrix):
    """Calculates Frequency-Weighted IoU from a global confusion matrix."""
    TP = np.diag(confusion_matrix)
    gt_pixels_per_class = np.sum(confusion_matrix, axis=1)
    pred_pixels_per_class = np.sum(confusion_matrix, axis=0)
    
    total_pixels = np.sum(gt_pixels_per_class)
    if total_pixels == 0:
        return 0.0

    union = gt_pixels_per_class + pred_pixels_per_class - TP
    iou_per_class = np.divide(TP, union, out=np.zeros_like(TP, dtype=float), where=union != 0)
    
    weights = gt_pixels_per_class / total_pixels
    return np.sum(weights * iou_per_class)

def rgb_to_class_indices(rgb_img, color_dict):
    """Maps an RGB image to a 2D array of class indices."""
    h, w, _ = rgb_img.shape
    idx_img = np.zeros((h, w), dtype=np.uint8)
    for class_idx, color in enumerate(color_dict.values()):
        mask = np.all(rgb_img == color, axis=-1)
        idx_img[mask] = class_idx
    return idx_img
# ==========================================

# Start overall timer
start_time_total = time.time()

# 1. Locate Local Dataset
print("Locating local UAVid dataset...")
images_dir = r"C:\Users\x3non\Desktop\q3 project y2\seq67\Images"
labels_dir = r"C:\Users\x3non\Desktop\q3 project y2\seq67\Labels"

image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
label_files = sorted(glob.glob(os.path.join(labels_dir, "*.png")))

if not image_files or len(image_files) != len(label_files):
    raise ValueError(f"Found {len(image_files)} images and {len(label_files)} labels.")

print(f"Found {len(image_files)} image-label pairs. Initializing models...")

uavid_gt_colors = {
    "building":            [128, 0, 0],
    "road":                [128, 64, 128],
    "tree":                [0, 128, 0],
    "low vegetation":      [128, 128, 0],
    "background clutter":  [0, 0, 0],
    "human":               [64, 64, 0],
    "car":                 [192, 0, 192]
}

class_names = list(uavid_gt_colors.keys())
num_classes = len(class_names)
color_array = np.array(list(uavid_gt_colors.values()), dtype=np.uint8)

# 2. Load Hugging Face Model (OPTIMIZED)
model_load_start = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading SAM3 model...")
model = Sam3Model.from_pretrained(
    "facebook/sam3", 
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa" # Changed from flash_attention_2
).to(device)

# Speed Optimization 2: Compile model (requires PyTorch 2.0+)
print("Compiling model for faster inference (this may take a moment)...")

processor = Sam3Processor.from_pretrained("facebook/sam3")
model_load_time = time.time() - model_load_start
print(f"Model loaded and compiled in {model_load_time:.2f}s")

# Build color index once for scoring
categories, colour_to_idx = build_colour_index(CATEGORY_COLOURS, merge_cars=True, merge_vegetation=False)

# Track metrics and timing
total_miou = 0.0
global_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
image_times = []
all_prompts = list(uavid_gt_colors.keys())

# 3. Batch Inference Loop
for img_idx, (img_path, lbl_path) in enumerate(zip(image_files, label_files)):
    img_start_time = time.time()  
    
    img_name = os.path.basename(img_path)
    print(f"\n[{img_idx + 1}/{len(image_files)}] Processing {img_name}...")

    image_pil = Image.open(img_path).convert("RGB")
    image_cv2 = np.array(image_pil)
    H, W, _ = image_cv2.shape

    global_scores = np.zeros((num_classes, H, W), dtype=np.float32)

    tile_size = 1024
    stride = 1024
    
    def get_tiles(width, height, tile_size, stride):
        tiles = []
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                x1, y1 = x, y
                x2, y2 = min(x + tile_size, width), min(y + tile_size, height)
                if x2 - x1 < tile_size: x1 = max(0, width - tile_size)
                if y2 - y1 < tile_size: y1 = max(0, height - tile_size)
                tiles.append((x1, y1, x2, y2))
        return list(set(tiles))

    tiles = get_tiles(W, H, tile_size, stride)
    print(f"  Processing {len(tiles)} tiles...")
    
    inference_start = time.time()  
    
    with torch.inference_mode():
        for tile_idx, (x1, y1, x2, y2) in enumerate(tiles):
            tile_pil = image_pil.crop((x1, y1, x2, y2))
            
            # Speed Optimization 3: Batch all 7 prompts together
            images_batch = [tile_pil] * num_classes
            
            inputs = processor(images=images_batch, text=all_prompts, return_tensors="pt").to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            
            outputs = model(**inputs)
            
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.35,       # Tweak these later for mIoU/FWIoU tuning
                mask_threshold=0.35,
                target_sizes=inputs.get("original_sizes").tolist()
            )
            
            # Process results for each class in the batch
            for class_idx, class_result in enumerate(results):
                masks = class_result["masks"].cpu().to(torch.float32).numpy()
                
                if len(masks) > 0:
                    scores = class_result.get("scores", torch.ones(len(masks))).cpu().to(torch.float32).numpy()
                    
                    weighted_masks = masks * scores[:, None, None]
                    best_tile_scores = np.max(weighted_masks, axis=0) 
                    
                    global_scores[class_idx, y1:y2, x1:x2] = np.maximum(
                        global_scores[class_idx, y1:y2, x1:x2], 
                        best_tile_scores
                    )
    
    inference_time = time.time() - inference_start
    print(f"  Inference time: {inference_time:.2f}s")

    # Resolve overlaps 
    overlay = np.zeros_like(image_cv2, dtype=np.uint8)
    best_class_indices = np.argmax(global_scores, axis=0)
    has_prediction = np.max(global_scores, axis=0) > 0
    overlay[has_prediction] = color_array[best_class_indices[has_prediction]]

    # FWIoU Calculations 
    gt_cv2 = np.array(Image.open(lbl_path).convert("RGB"))
    gt_indices = rgb_to_class_indices(gt_cv2, uavid_gt_colors)
    pred_indices = rgb_to_class_indices(overlay, uavid_gt_colors)
    
    image_cm = fast_hist(gt_indices.flatten(), pred_indices.flatten(), num_classes)
    image_fwiou = calculate_fwiou(image_cm)
    global_confusion_matrix += image_cm

    # Standard Scoring 
    pred_path = f"temp_pred_{img_name}"
    Image.fromarray(overlay).save(pred_path) 

    per_class, miou, *_ = evaluate_pair(lbl_path, pred_path, colour_to_idx, categories)
    
    print_results(per_class, miou, image_name=img_name)
    print(f"  --> Image FWIoU: {image_fwiou:.4f}")

    total_miou += miou
    os.remove(pred_path)
    
    img_time = time.time() - img_start_time
    image_times.append(img_time)
    print(f"  Total time for {img_name}: {img_time:.2f}s")
    
    avg_time_per_image = sum(image_times) / len(image_times)
    estimated_remaining = avg_time_per_image * (len(image_files) - (img_idx + 1))
    print(f"  Estimated time remaining: {timedelta(seconds=int(estimated_remaining))}")

# Final Global Metrics
final_fwiou = calculate_fwiou(global_confusion_matrix)
total_time = time.time() - start_time_total

print(f"\n{'='*50}")
print(f"=== BATCH COMPLETE ===")
print(f"{'='*50}")
print(f"Average mIoU (Mean of image means): {total_miou / len(image_files):.4f}")
print(f"Global FWIoU (Frequency-Weighted):  {final_fwiou:.4f}")
print(f"\nTiming Summary:")
print(f"  Model load/compile: {model_load_time:.2f}s")
print(f"  Total processing: {total_time:.2f}s ({timedelta(seconds=int(total_time))})")
print(f"  Average per image: {sum(image_times) / len(image_times):.2f}s")
print(f"  Fastest image: {min(image_times):.2f}s")
print(f"  Slowest image: {max(image_times):.2f}s")
print(f"{'='*50}")