import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
from transformers import Sam3Model, Sam3Processor
 
# --- magic bug fix by gemini ---
import huggingface_hub.dataclasses
huggingface_hub.dataclasses.type_validator = lambda *args, **kwargs: None
# ---- thank you mr clanker -----
 
# =============================================================================
# 1. CONFIG
# =============================================================================
 
image_path  = r"C:\Users\x3non\Desktop\q3 project y2\000000.png"
labels_path = r"C:\Users\x3non\Desktop\q3 project y2\000000labels.png"
 
PADDING             = 10   # pixels of context added around each bbox crop
MIN_COMPONENT_AREA  = 500  # minimum blob area (px²) to keep from pass 1
MIN_BOX_SIDE        = 10   # minimum bbox side length (px) to keep
 
uavid_gt_colors = {
    "building":                   [128, 0, 0],

    "road":                       [128, 64, 128],

    "tree":                       [0, 128, 0],
    "tree canopy":                [0, 128, 0],

    "grass":                      [128, 128, 0],

    "general background clutter": [0, 0, 0],
    "sidewalk":                   [0, 0, 0],

    "person":                     [64, 64, 0],
    "human":                      [64, 64, 0],
    "pedestrian":                 [64, 64, 0],


    "car":                        [192, 0, 192],
    "van":                        [192, 0, 192]
}
 
# =============================================================================
# 2. LOAD MODEL
# =============================================================================
 
print("Loading SAM3...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model     = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16).to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")
 
# =============================================================================
# 3. LOAD IMAGES
# =============================================================================
 
image_pil  = Image.open(image_path).convert("RGB")
image_cv2  = np.array(image_pil)
labels_pil = Image.open(labels_path).convert("RGB")
labels_cv2 = np.array(labels_pil)
overlay    = np.zeros_like(image_cv2, dtype=np.uint8)
 
img_w, img_h = image_pil.size
 
# =============================================================================
# 4. HELPER: extract bounding boxes from a binary mask via connected components
# =============================================================================
 
def masks_to_boxes(combined_mask):
    labeled, num_features = ndimage.label(combined_mask)
    boxes = []
    for comp_id in range(1, num_features + 1):
        component = labeled == comp_id
        if component.sum() < MIN_COMPONENT_AREA:
            continue
        rows = np.any(component, axis=1)
        cols = np.any(component, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        if (x2 - x1) < MIN_BOX_SIDE or (y2 - y1) < MIN_BOX_SIDE:
            continue
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes
 
def to_bfloat16(inputs):
    return {
        k: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v
        for k, v in inputs.items()
    }
 
# =============================================================================
# 5. PASS 1: text-prompted segmentation → coarse masks → bounding boxes
# =============================================================================
 
print("\nPass 1: text-prompted segmentation...")
coarse_results = {}  # color tuple → list of [x1, y1, x2, y2]
 
with torch.inference_mode():
    for prompt_text, color in uavid_gt_colors.items():
        if prompt_text == "general background clutter" or prompt_text == "sidewalk":
            print(f"  [Pass 1] Skipping '{prompt_text}' (default background)...")
            continue

        color_key = tuple(color)
 
        print(f"  [Pass 1] Segmenting: {prompt_text}...")
        inputs = processor(images=image_pil, text=prompt_text, return_tensors="pt").to(device)
        inputs = to_bfloat16(inputs)
 
        outputs = model(**inputs)
 
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.4,
            mask_threshold=0.4,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
 
        masks = results["masks"]
        if results.get("masks") is None or len(results["masks"]) == 0:
            print(f"    → No masks found")
            coarse_results[color_key] = []
            continue
 
        combined_mask = np.any(masks.cpu().numpy(), axis=0)
        boxes = masks_to_boxes(combined_mask)
        coarse_results[color_key] = boxes
        print(f"    → {len(boxes)} component(s) found")

# =============================================================================
# 6. PASS 2: High-Fidelity Crop-and-Resegment
# =============================================================================

print("\nPass 2: High-fidelity crop-and-resegment...")

with torch.inference_mode():
    for prompt_text, color in uavid_gt_colors.items():
        color_key = tuple(color)
        boxes = coarse_results.get(color_key, [])

        if not boxes:
            continue

        print(f"  [Pass 2] Refining '{prompt_text}' ({len(boxes)} crop(s))...")
        color_arr = np.array(color, dtype=np.uint8)

        for (x1, y1, x2, y2) in boxes:
            # 1. Pad and clamp to image bounds
            cx1 = max(0, x1 - PADDING)
            cy1 = max(0, y1 - PADDING)
            cx2 = min(img_w, x2 + PADDING)
            cy2 = min(img_h, y2 + PADDING)

            crop = image_pil.crop((cx1, cy1, cx2, cy2))
            crop_w = cx2 - cx1
            crop_h = cy2 - cy1

            # 2. Define a local bounding box strictly inside the crop's dimensions
            # We inset it slightly from the edges to give SAM boundary context
            local_box = [PADDING // 2, PADDING // 2, crop_w - (PADDING // 2), crop_h - (PADDING // 2)]
            
            # Format expected by HF: [batch_size=1, num_boxes=1, 4 coordinates]
            input_boxes = [[local_box]]

            # 3. Process the high-resolution crop
            inputs = processor(
                images=crop,
                input_boxes=input_boxes,
                return_tensors="pt"
            ).to(device)
            
            inputs = to_bfloat16(inputs)
            outputs = model(**inputs)

            # 4. Post-process the mask for this specific crop size
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.2,
                mask_threshold=0.2,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]

            masks = results.get("masks")
            if masks is not None and len(masks) > 0:
                # Get the highest confidence mask and convert to boolean
                mask_crop = masks[0].cpu().numpy().astype(bool)
                
                # 5. Paste the perfectly refined mask back into the global overlay
                overlay[cy1:cy2, cx1:cx2][mask_crop] = color_arr
            else:
                # Fallback: paint the coarse box if refinement completely fails
                overlay[y1:y2, x1:x2] = color_arr

# =============================================================================
# 7. VISUALISATION
# =============================================================================
 
print("\nGenerating visualisation...")
fig, axes = plt.subplots(1, 2, figsize=(24, 8))
 
axes[0].imshow(overlay)
axes[0].set_title("SAM3 Segmentation (two-pass)")
axes[0].axis("off")
 
axes[1].imshow(labels_cv2)
axes[1].set_title("Ground Truth Labels")
axes[1].axis("off")
 
plt.tight_layout()
plt.show()
 
# =============================================================================
# 8. SCORING
# =============================================================================
 
import sys, os
sys.path.append(r"C:\Users\x3non\OneDrive\Desktop\A03-AE2224")
from scoring_general import (
    CATEGORY_COLOURS, build_colour_index, evaluate_pair, print_results
)
 
pred_path = "predicted_segmentation.png"
Image.fromarray(overlay).save(pred_path)
 
categories, colour_to_idx = build_colour_index(
    CATEGORY_COLOURS, merge_cars=True, merge_vegetation=True
)
 
per_class, miou, _ = evaluate_pair(
    labels_path, pred_path, colour_to_idx, categories
)
 
print_results(per_class, miou, image_name="000000.png")
 
os.remove(pred_path)