import os
import glob
import torch
import numpy as np
import cv2
import kagglehub
from PIL import Image
from transformers import Sam3Model, Sam3Processor

# --- magic bug fix by gemini ---
import huggingface_hub.dataclasses
huggingface_hub.dataclasses.type_validator = lambda *args, **kwargs: None
# ---- thank you mr clanker -----

# Import evaluation functions
import sys
sys.path.append(r"C:\Users\x3non\OneDrive\Desktop\A03-AE2224")  
from scoring_general import CATEGORY_COLOURS, build_colour_index, evaluate_pair, print_results

# 1. Download and Locate Dataset
print("Downloading/Locating UAVid dataset...")
dataset_path = kagglehub.dataset_download("dasmehdixtr/uavid-v1")
print(f"Dataset root: {dataset_path}")

# Construct paths to seq16
# Note: Some dataset versions capitalize 'Images' and 'Labels'
images_dir = os.path.join(dataset_path, "uavid_val", "seq16", "Images")
labels_dir = os.path.join(dataset_path, "uavid_val", "seq16", "Labels")

# Get sorted lists of all images and labels to ensure they match up perfectly
image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
label_files = sorted(glob.glob(os.path.join(labels_dir, "*.png")))

if not image_files or len(image_files) != len(label_files):
    raise ValueError(f"Found {len(image_files)} images and {len(label_files)} labels. Check directory paths.")

print(f"Found {len(image_files)} image-label pairs. Initializing models...")

uavid_gt_colors = {
    "building":                   [128, 0, 0],

    "paved road":                 [128, 64, 128],

    "tree":                       [0, 128, 0],
    "tree canopy":                [0, 128, 0],

    "grass":                      [128, 128, 0],
    "bush":                       [128, 128, 0],
    "low vegetation":             [128, 128, 0],

    "general background clutter": [0, 0, 0],
    "sidewalk":                   [0, 0, 0],
    "public square":              [0, 0, 0],

    "person":                     [64, 64, 0],
    "human":                      [64, 64, 0],
    "pedestrian":                 [64, 64, 0],

    "car":                        [192, 0, 192],
    "van":                        [192, 0, 192]
}

class_names = list(uavid_gt_colors.keys())
num_classes = len(class_names)
color_array = np.array(list(uavid_gt_colors.values()), dtype=np.uint8)

# 2. Load Hugging Face Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16).to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# Build color index once for scoring
categories, colour_to_idx = build_colour_index(CATEGORY_COLOURS, merge_cars=True, merge_vegetation=False)

# Track overall mIoU
total_miou = 0.0

# 3. Batch Inference Loop
for img_idx, (img_path, lbl_path) in enumerate(zip(image_files, label_files)):
    img_name = os.path.basename(img_path)
    print(f"\n[{img_idx + 1}/{len(image_files)}] Processing {img_name}...")

    # Load images
    image_pil = Image.open(img_path).convert("RGB")
    image_cv2 = np.array(image_pil)
    H, W, _ = image_cv2.shape

    # Reset global scores for the new image
    global_scores = np.zeros((num_classes, H, W), dtype=np.float32)

    # Tiling Setup
    tile_size = 1024
    stride = 512  
    
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

    with torch.inference_mode():
        for tile_idx, (x1, y1, x2, y2) in enumerate(tiles):
            tile_pil = image_pil.crop((x1, y1, x2, y2))
            
            for class_idx, (prompt_text, _) in enumerate(uavid_gt_colors.items()):
                inputs = processor(images=tile_pil, text=prompt_text, return_tensors="pt").to(device)
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                
                outputs = model(**inputs)
                
                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=inputs.get("original_sizes").tolist()
                )[0]
                
                # FIXED: Cast to float32 before numpy conversion
                masks = results["masks"].cpu().to(torch.float32).numpy()
                
                if len(masks) > 0:
                    # FIXED: Cast to float32 before numpy conversion
                    scores = results.get("scores", torch.ones(len(masks))).cpu().to(torch.float32).numpy()
                    
                    weighted_masks = masks * scores[:, None, None]
                    best_tile_scores = np.max(weighted_masks, axis=0) 
                    
                    global_scores[class_idx, y1:y2, x1:x2] = np.maximum(
                        global_scores[class_idx, y1:y2, x1:x2], 
                        best_tile_scores
                    )

    # Resolve overlaps for this image
    overlay = np.zeros_like(image_cv2, dtype=np.uint8)
    best_class_indices = np.argmax(global_scores, axis=0)
    has_prediction = np.max(global_scores, axis=0) > 0
    overlay[has_prediction] = color_array[best_class_indices[has_prediction]]

    # Scoring Integration
    pred_path = f"temp_pred_{img_name}"
    Image.fromarray(overlay).save(pred_path) 

    per_class, miou, _ = evaluate_pair(lbl_path, pred_path, colour_to_idx, categories)
    print_results(per_class, miou, image_name=img_name)
    
    total_miou += miou
    os.remove(pred_path)

print(f"\n=== BATCH COMPLETE ===")
print(f"Average mIoU across all 70 images: {total_miou / len(image_files):.4f}")