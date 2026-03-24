# v1 comparison to results and scoring integration

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- magic bug fix by gemini ---
# This intercepts the Hugging Face library and turns off the strict type 
# checker that is crashing because of their 'initializer_range' typo.
import huggingface_hub.dataclasses
huggingface_hub.dataclasses.type_validator = lambda *args, **kwargs: None
# ---- thank you mr clanker -----

from PIL import Image
from transformers import Sam3Model, Sam3Processor

# 1. Setup
image_path = r"C:\Users\x3non\Desktop\q3 project y2\000000.png"

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

# 2. Load Hugging Face Model
print("Loading Hugging Face SAM3...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# We load in bfloat16 to save VRAM and increase speed
model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16).to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

# 3. Load Image
image_pil = Image.open(image_path).convert("RGB")
image_cv2 = np.array(image_pil)
labels_path = r"C:\Users\x3non\Desktop\q3 project y2\000000labels.png"
labels_pil = Image.open(labels_path).convert("RGB")
labels_cv2 = np.array(labels_pil)
overlay = np.zeros_like(image_cv2, dtype=np.uint8)


print("Starting inference...")
# 4. Inference Loop
with torch.inference_mode():
    for prompt_text, color in uavid_gt_colors.items():
        print(f"Segmenting: {prompt_text}...")
        
        # Prepare inputs for the model
        inputs = processor(images=image_pil, text=prompt_text, return_tensors="pt").to(device)
        
        # Convert pixel values to match the model's bfloat16 precision
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        
        # Run forward pass
        outputs = model(**inputs)
        
        # Post-process the outputs to get the masks back to the original image size
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.2,
            mask_threshold=0.2,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        # Apply the color if objects were found
        masks = results["masks"]
        if len(masks) > 0:
            # Move to CPU, convert to numpy, and flatten multiple instances into a single mask
            masks_data = masks.cpu().numpy()
            combined_mask = np.any(masks_data, axis=0) 
            
            color_rgb = np.array(color, dtype=np.uint8)
            overlay[combined_mask] = color_rgb

# 5. Display Side-by-Side Comparison
print("Generating visualization...")
fig, axes = plt.subplots(1, 2, figsize=(24, 8))

# SAM3 Segmentation
axes[0].imshow(overlay)
axes[0].set_title("SAM3 Segmentation")
axes[0].axis("off")

# Ground truth labels
axes[1].imshow(labels_cv2)
axes[1].set_title("Ground Truth Labels")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# =============================================================================
# SCORING INTEGRATION
# =============================================================================

# Import evaluation functions from scoring_general.py
import sys
sys.path.append(r"C:\Users\x3non\OneDrive\Desktop\A03-AE2224")  # Adjust path if needed
from scoring_general import (
    CATEGORY_COLOURS, build_colour_index, evaluate_pair, print_results
)

# Save the predicted overlay as a temporary RGB image for evaluation
pred_path = "predicted_segmentation.png"
Image.fromarray(overlay).save(pred_path)  # PIL expects RGB

# Build color index
categories, colour_to_idx = build_colour_index(
    CATEGORY_COLOURS, merge_cars=True, merge_vegetation=True
)

# Evaluate the prediction against ground truth
per_class, miou, _ = evaluate_pair(
    labels_path, pred_path, colour_to_idx, categories
)

# Print the results
print_results(per_class, miou, image_name="000000.png")

# Optional: Clean up the temporary file
import os
os.remove(pred_path)