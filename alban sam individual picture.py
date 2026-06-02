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
from releases.support.scoring_general import CATEGORY_COLOURS, build_colour_index, evaluate_pair, print_results

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
print("Targeting single UAVid image...")

# Define the exact single image and label path
target_image_path = r"C:\Users\x3non\Desktop\q3 project y2\uavid_train\seq1\Images\000800.png"
target_label_path = r"C:\Users\x3non\Desktop\q3 project y2\uavid_train\seq1\Labels\000800.png"

image_files = [target_image_path]
label_files = [target_label_path]

if not os.path.exists(target_image_path) or not os.path.exists(target_label_path):
    raise ValueError("Could not find the specified image or label path. Please check the directories.")

print(f"Found image. Initializing models...")

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

model = Sam3Model.from_pretrained(
    "facebook/sam3", 
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    local_files_only=True
).to(device)

processor = Sam3Processor.from_pretrained(
    "facebook/sam3",
    local_files_only=True
)
model_load_time = time.time() - model_load_start
print(f"Model loaded and compiled in {model_load_time:.2f}s")

# Build color index once for scoring
categories, colour_to_idx = build_colour_index(CATEGORY_COLOURS, merge_cars=True, merge_vegetation=False)

# Track metrics and timing
total_miou = 0.0
global_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
image_times = []
all_prompts = list(uavid_gt_colors.keys())

# 3. Single Inference Run
for img_idx, (img_path, lbl_path) in enumerate(zip(image_files, label_files)):
    img_start_time = time.time()  
    
    img_name = os.path.basename(img_path)
    print(f"\nProcessing {img_name}...")

    image_pil = Image.open(img_path).convert("RGB")
    image_cv2 = np.array(image_pil)
    H, W, _ = image_cv2.shape

    global_scores = np.zeros((num_classes, H, W), dtype=np.float32)

    tile_size = 1024
    stride = 768
    
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
                threshold=0.4,
                mask_threshold=0.4,
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

    # Standard Scoring and Saving
    pred_path = f"output_pred_{img_name}"
    Image.fromarray(overlay).save(pred_path) 
    print(f"  Saved final mask to: {os.path.abspath(pred_path)}")

    per_class, miou, *_ = evaluate_pair(lbl_path, pred_path, colour_to_idx, categories)
    
    print_results(per_class, miou, image_name=img_name)
    print(f"  --> Image FWIoU: {image_fwiou:.4f}")

    total_miou += miou
    # os.remove(pred_path) <-- Removed to keep the file
    
    img_time = time.time() - img_start_time
    image_times.append(img_time)
    print(f"  Total time for {img_name}: {img_time:.2f}s")

# Final Global Metrics
final_fwiou = calculate_fwiou(global_confusion_matrix)

# Calculate Precision, Recall, and F1-Score
TP = np.diag(global_confusion_matrix)
GT_total = np.sum(global_confusion_matrix, axis=1)
Pred_total = np.sum(global_confusion_matrix, axis=0)

precision_per_class = np.divide(TP, Pred_total, out=np.zeros_like(TP, dtype=float), where=Pred_total != 0)
recall_per_class = np.divide(TP, GT_total, out=np.zeros_like(TP, dtype=float), where=GT_total != 0)

f1_denominator = precision_per_class + recall_per_class
f1_per_class = np.divide(2 * (precision_per_class * recall_per_class), f1_denominator, out=np.zeros_like(TP, dtype=float), where=f1_denominator != 0)

mean_precision = np.mean(precision_per_class)
mean_recall = np.mean(recall_per_class)
mean_f1 = np.mean(f1_per_class)

total_time = time.time() - start_time_total

print(f"\n{'='*50}")
print(f"=== PROCESSING COMPLETE ===")
print(f"{'='*50}")
print(f"Image mIoU: {total_miou:.4f}")
print(f"Image FWIoU:  {final_fwiou:.4f}")
print(f"Image Precision (Macro Mean): {mean_precision:.4f}")
print(f"Image Recall (Macro Mean):    {mean_recall:.4f}")
print(f"Image F1-Score (Macro Mean):  {mean_f1:.4f}")
print(f"\n--- Per-Class Metrics ---")
for i, cls_name in enumerate(class_names):
    print(f"{cls_name.ljust(20)} | Precision: {precision_per_class[i]:.4f} | Recall: {recall_per_class[i]:.4f} | F1: {f1_per_class[i]:.4f}")

print(f"\nTiming Summary:")
print(f"  Model load/compile: {model_load_time:.2f}s")
print(f"  Total processing: {total_time:.2f}s ({timedelta(seconds=int(total_time))})")
print(f"{'='*50}")