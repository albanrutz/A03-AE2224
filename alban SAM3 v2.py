import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Sam3Model, Sam3Processor

# --- magic bug fix by gemini ---
import huggingface_hub.dataclasses
huggingface_hub.dataclasses.type_validator = lambda *args, **kwargs: None
# ---- thank you mr clanker -----

# 1. Setup & Configuration
image_path = r"C:\Users\x3non\Desktop\q3 project y2\000000.png"
labels_path = r"C:\Users\x3non\Desktop\q3 project y2\000000labels.png"

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

# 2. Load Hugging Face Model
print("Loading Hugging Face SAM3...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16).to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# 3. Load Image & Setup Global Arrays
image_pil = Image.open(image_path).convert("RGB")
image_cv2 = np.array(image_pil)
labels_cv2 = np.array(Image.open(labels_path).convert("RGB"))
H, W, _ = image_cv2.shape

class_names = list(uavid_gt_colors.keys())
num_classes = len(class_names)
# 3D array: [class_index, height, width] storing confidence scores
global_scores = np.zeros((num_classes, H, W), dtype=np.float32)

# 4. Tiling Setup (50% Overlap for Cross-Patch Objects)
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

# 5. Inference Loop
print(f"Starting inference across {len(tiles)} tiles (4K integration)...")
with torch.inference_mode():
    for tile_idx, (x1, y1, x2, y2) in enumerate(tiles):
        print(f"Processing tile {tile_idx+1}/{len(tiles)}...")
        tile_pil = image_pil.crop((x1, y1, x2, y2))
        
        for class_idx, (prompt_text, _) in enumerate(uavid_gt_colors.items()):
            inputs = processor(images=tile_pil, text=prompt_text, return_tensors="pt").to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            
            outputs = model(**inputs)
            
            # Optimized Thresholds for tighter bounds
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.3,
                mask_threshold=0.3,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
            
            masks = results["masks"].cpu().to(torch.float32).numpy()
            
            if len(masks) > 0:
                # Extract scores; default to 1.0 if the processor doesn't return them
                scores = results.get("scores", torch.ones(len(masks))).cpu().to(torch.float32).numpy()
                
                # Weight binary masks by their confidence score
                weighted_masks = masks.astype(np.float32) * scores[:, None, None]
                
                # Flatten multiple instances inside this specific tile
                best_tile_scores = np.max(weighted_masks, axis=0) 
                
                # Update global map: np.maximum stitches overlapping patch seams based on confidence
                global_scores[class_idx, y1:y2, x1:x2] = np.maximum(
                    global_scores[class_idx, y1:y2, x1:x2], 
                    best_tile_scores
                )

# 6. Resolve Overlaps & Map Colors
print("Resolving inter-class overlaps...")
overlay = np.zeros_like(image_cv2, dtype=np.uint8)
color_array = np.array(list(uavid_gt_colors.values()), dtype=np.uint8)

# Argmax resolves class priority (e.g., pedestrian standing on a road)
best_class_indices = np.argmax(global_scores, axis=0)
has_prediction = np.max(global_scores, axis=0) > 0

overlay[has_prediction] = color_array[best_class_indices[has_prediction]]

# 7. Visualization
print("Generating visualization...")
fig, axes = plt.subplots(1, 2, figsize=(24, 8))

axes[0].imshow(overlay)
axes[0].set_title("SAM 3 Output (4K Tiled & Smoothed)")
axes[0].axis("off")

axes[1].imshow(labels_cv2)
axes[1].set_title("Ground Truth Labels")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# =============================================================================
# SCORING INTEGRATION
# =============================================================================
print("Calculating mIoU metrics...")
import sys
sys.path.append(r"C:\Users\x3non\OneDrive\Desktop\A03-AE2224")  
from scoring_general import CATEGORY_COLOURS, build_colour_index, evaluate_pair, print_results

pred_path = "predicted_segmentation.png"
Image.fromarray(overlay).save(pred_path) 

categories, colour_to_idx = build_colour_index(CATEGORY_COLOURS, merge_cars=True, merge_vegetation=True)
per_class, miou, _ = evaluate_pair(labels_path, pred_path, colour_to_idx, categories)

print_results(per_class, miou, image_name="000000.png")

import os
os.remove(pred_path)