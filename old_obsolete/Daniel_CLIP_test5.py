import torch
import clip
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torch.amp import autocast # Required for the RTX 4070 VRAM optimization

# ==========================================
# 1. CORE FUNCTIONS 
# ==========================================

def sliding_window_segmentation(image_rgb, labels, patch_size=224):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Convert the unified matrix back to PIL for cropping logic
    original_image = Image.fromarray(image_rgb)
    W, H = original_image.size
    cols, rows = W // patch_size, H // patch_size

    segmentation_map = np.zeros((rows, cols), dtype=np.int32)

    text_tokens = clip.tokenize(labels).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    print(f"[Grid Map] Extracting {rows * cols} patches of size {patch_size}x{patch_size}...")

    patches, coordinates = [], []
    for r in range(rows):
        for c in range(cols):
            left, upper = c * patch_size, r * patch_size
            right, lower = left + patch_size, upper + patch_size
            patch = original_image.crop((left, upper, right, lower))
            patches.append(preprocess(patch))
            coordinates.append((r, c))

    batch_tensor = torch.stack(patches).to(device)
    batch_size = 64
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(batch_tensor), batch_size), desc="Classifying Patches"):
            chunk = batch_tensor[i : i + batch_size]
            image_features = model.encode_image(chunk)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ text_features.T
            predictions.extend(similarities.argmax(dim=-1).cpu().numpy())

    for (r, c), winner_idx in zip(coordinates, predictions):
        segmentation_map[r, c] = winner_idx

    # Scale the grid back to the unified image dimensions
    grid_map_resized = cv2.resize(
        segmentation_map, 
        (W, H), 
        interpolation=cv2.INTER_NEAREST
    )
    return grid_map_resized

def extract_geometric_features(image_rgb, sam_checkpoint="sam_vit_b_01ec64.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Empty VRAM cache before the heavy pass
    if device == "cuda":
        torch.cuda.empty_cache()
        
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam, 
        points_per_side=32, 
        points_per_batch=16, # RTX 4070 VRAM Throttle
        pred_iou_thresh=0.82, 
        stability_score_thresh=0.92, 
        crop_n_layers=1, 
        crop_n_points_downscale_factor=2, 
        min_mask_region_area=100
    )
    
    print("[SAM] Computing geometric boundaries on GPU...")
    
    # Hardware-level memory compression for 12GB GPUs
    with torch.inference_mode(), autocast(device_type="cuda", dtype=torch.float16):
        masks = mask_generator.generate(image_rgb)
        
    print(f"[SAM] Identified {len(masks)} distinct features.")
    
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    extracted_features = []
    
    for mask_data in masks:
        bool_mask = mask_data['segmentation']
        isolated_feature = np.zeros_like(image_rgb)
        isolated_feature[bool_mask] = image_rgb[bool_mask]
        
        x, y, w, h = [int(v) for v in mask_data['bbox']]
        extracted_features.append({
            'cropped_image': isolated_feature[y:y+h, x:x+w],
            'full_mask': bool_mask,
            'area': mask_data['area']
        })
        
    return extracted_features

def classify_sam_features(image_rgb, extracted_features, labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    text_tokens = clip.tokenize(labels).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    print(f"[SAM-CLIP] Preprocessing {len(extracted_features)} features...")
    patch_tensors = []
    for feature in extracted_features:
        pil_img = Image.fromarray(feature['cropped_image'])
        patch_tensors.append(preprocess(pil_img).unsqueeze(0).to(device).type(model.dtype))

    batch_tensor = torch.cat(patch_tensors)
    
    predictions = []
    probabilities = []

    print("[SAM-CLIP] Computing probabilities...")
    with torch.no_grad():
        logit_scale = model.logit_scale.exp()
        
        for i in range(0, len(batch_tensor), 64):
            chunk = batch_tensor[i : i + 64]
            image_features = model.encode_image(chunk)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            similarities = image_features @ text_features.T
            probs = (similarities * logit_scale).softmax(dim=-1)
            max_probs, winners = probs.max(dim=-1)
            
            predictions.extend(winners.cpu().numpy())
            probabilities.extend(max_probs.cpu().numpy())

    return predictions, probabilities

def panoptic_fusion(grid_map, sam_features, sam_predictions, sam_probs, target_classes, conf_thresh=0.60, max_sam_area=15000):
    fused_map = grid_map.copy()
    objects_added = 0 
    
    for feature, predicted_idx, prob in zip(sam_features, sam_predictions, sam_probs):
        if feature['area'] < max_sam_area:
            if predicted_idx in target_classes:
                if prob > conf_thresh:
                    fused_map[feature['full_mask']] = predicted_idx
                    objects_added += 1
                    
    print(f"[Fusion] Filtered {len(sam_features)} SAM polygons down to {objects_added} high-confidence objects.")
    return fused_map

def visualize_final_fusion(image_rgb, fused_map, labels):
    colors = plt.cm.get_cmap('tab10', len(labels))
    cmap = ListedColormap(colors.colors)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    ax[0].imshow(image_rgb)
    ax[0].set_title("Original Aligned UAV Image")
    ax[0].axis('off')

    ax[1].imshow(image_rgb)
    ax[1].imshow(fused_map, cmap=cmap, vmin=0, vmax=len(labels)-1, alpha=0.6)
    ax[1].set_title("Panoptic Fusion (Filtered)")
    ax[1].axis('off')

    patches_legend = [mpatches.Patch(color=colors(i), label=labels[i]) for i in range(len(labels))]
    ax[1].legend(handles=patches_legend, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

# ==========================================
# 2. MASTER EXECUTION ENGINE
# ==========================================

if __name__ == "__main__":
    
    # Setup your paths and parameters
    image_path = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1\Images\file14-2.png"
    patch_size = 224

    uavid_labels = [
        "aerial view of a building",            # 0
        "aerial view of road with no cars",     # 1
        "aerial view of a tree",                # 2  <-- TARGET
        "aerial view of grass yard",            # 3
        "aerial view of low vegetation bushes", # 4
        "aerial view of background clutter",    # 5
        "photo of a car",                       # 6  <-- TARGET
        "photo of a human"                      # 7  <-- TARGET
    ]
    
    thing_indices = [2, 6, 7] 

    print("=== Pipeline Started ===")

    # --- Step 1: Establish the Geometric Baseline ---
    original_img = Image.open(image_path).convert("RGB")
    W, H = original_img.size
    cropped_W, cropped_H = (W // patch_size) * patch_size, (H // patch_size) * patch_size
    master_image_rgb = np.array(original_img.crop((0, 0, cropped_W, cropped_H)))

    # --- Step 2: Extract Continuous Terrain (Stuff) ---
    grid_map = sliding_window_segmentation(master_image_rgb, uavid_labels, patch_size)

    # --- Step 3: Extract Discrete Objects (Things) ---
    sam_features = extract_geometric_features(master_image_rgb)
    sam_preds, sam_probs = classify_sam_features(master_image_rgb, sam_features, uavid_labels)

    # --- Step 4: Mathematical Fusion (Z-Buffering) ---
    final_panoptic_map = panoptic_fusion(
        grid_map, 
        sam_features, 
        sam_preds, 
        sam_probs, 
        target_classes=thing_indices, 
        conf_thresh=0.60, 
        max_sam_area=15000
    )

    # --- Step 5: Visualization ---
    visualize_final_fusion(master_image_rgb, final_panoptic_map, uavid_labels)
    print("=== Pipeline Complete ===")