"""
BlackMagiCLIP: Multi-Scale Bayesian Panoptic Engine
===================================================
This script implements a zero-shot semantic segmentation pipeline using a frozen CLIP (RN50) model.
It utilizes:
1. SpatialLayerHooks for dense Grad-CAM feature extraction from hidden layers.
2. Multi-Scale Patching with interpolation-free padding.
3. A Scale-Class Weight Matrix to natively handle the "Thing vs. Stuff" spatial dichotomy.
4. Temperature Softening to mathematically reduce hyper-confident fragmentation.
"""

import torch
import torch.nn.functional as F
import clip
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm
import os
from scoring_general import save_and_evaluate_single_image

# =============================================================================
# --- 1. HARDWARE & CONTEXT INITIALIZATION ---
# =============================================================================

if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.set_device(0)
    # Warm up the GPU matrix cores
    _ = torch.ones(1, device="cuda") @ torch.ones(1, device="cuda")

# =============================================================================
# --- 2. THE SPATIAL HOOK (GRAD-CAM ENGINE) ---
# =============================================================================

class SpatialLayerHook:
    def __init__(self, module):
        self.activations = None
        self.gradients = None
        self.hook = module.register_forward_hook(self.hook_fn)
        
    def hook_fn(self, module, input, output):
        self.activations = output
        self.hook_grad = output.register_hook(self.save_gradient)
        
    def save_gradient(self, grad):
        self.gradients = grad
        
    def close(self):
        self.hook.remove()

# =============================================================================
# --- 3. HELPER FUNCTIONS ---
# =============================================================================

def segment_mask_to_rgb(global_seg_map, labels_order):
    """Converts a 2D segmentation map of integer indices into a 3D RGB image array."""
    uavid_gt_colors = {
        "building":    [128, 0, 0],
        "road":        [128, 64, 128],
        "tree":        [0, 128, 0],
        "low_veg":     [128, 128, 0],
        "clutter":     [0, 0, 0],
        "car":         [192, 0, 192],
        "human":       [64, 64, 0]
    }
    num_classes = len(labels_order)
    color_matrix = np.zeros((num_classes, 3), dtype=np.uint8)
    for i, label in enumerate(labels_order):
        if label in uavid_gt_colors:
            color_matrix[i] = uavid_gt_colors[label]
        else:
            color_matrix[i] = [255, 255, 255]
            
    return color_matrix[global_seg_map]

# =============================================================================
# --- 4. THE CORE ARCHITECTURE ---
# =============================================================================

def exact_multi_scale_ensemble_matrix(image_path, clip_prompts, mapping_keys, scale_class_matrix, temperature=1.0, sw_plot=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # CRITICAL: RN50 is required for the SpatialLayerHook (CNN Architecture)
    model, preprocess = clip.load("RN50", device=device)
    model.eval()
    
    # Attach the hook to Layer 4
    hook = SpatialLayerHook(model.visual.layer4)

    original_image = Image.open(image_path).convert("RGB")
    W_orig, H_orig = original_image.size
    num_classes = len(clip_prompts)
    
    # Initialize the 3D Probability Tensor at EXACT native resolution
    fused_prob_tensor = np.zeros((H_orig, W_orig, num_classes), dtype=np.float32)

    text_tokens = clip.tokenize(clip_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Extract the patch scales from the matrix keys and sort descending
    patch_scales = sorted(list(scale_class_matrix.keys()), reverse=True)

    for p_size in patch_scales:
        print(f"\n--- Processing Scale: {p_size}x{p_size} (Grad-CAM Enabled) ---")
        
        # 1. Extract the specific class weights for THIS scale
        scale_weights_dict = scale_class_matrix[p_size]
        current_weight_array = np.array(
            [scale_weights_dict.get(k, 1.0) for k in mapping_keys], 
            dtype=np.float32
        )
        
        # 2. Calculate necessary padding for interpolation-free extraction
        pad_w = (p_size - (W_orig % p_size)) % p_size
        pad_h = (p_size - (H_orig % p_size)) % p_size
        padded_w, padded_h = W_orig + pad_w, H_orig + pad_h
        
        padded_img = Image.new("RGB", (padded_w, padded_h), color=(0, 0, 0))
        padded_img.paste(original_image, (0, 0))
        
        cols, rows = padded_w // p_size, padded_h // p_size
        padded_scale_tensor = np.zeros((padded_h, padded_w, num_classes), dtype=np.float32)

        patches, boxes = [], []
        
        for r in range(rows):
            for c in range(cols):
                left, upper = c * p_size, r * p_size
                right, lower = left + p_size, upper + p_size
                
                patch = padded_img.crop((left, upper, right, lower))
                patches.append(preprocess(patch))
                boxes.append((upper, lower, left, right))

        batch_tensor = torch.stack(patches).to(device)
        
        # Batch size lowered to 32 to accommodate 7 backpropagation passes per patch
        batch_size = 32 
        all_dense_probs = []
        
        # 3. Inference & Batched Backpropagation
        for i in tqdm(range(0, len(batch_tensor), batch_size), desc=f"Evaluating Patches"):
            chunk = batch_tensor[i : i + batch_size].type(model.dtype)
            chunk.requires_grad = True # Mandatory for backprop
            
            # Forward Pass
            image_features = model.encode_image(chunk).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Algorithmic Fix: Temperature Softening applied here
            raw_logits = model.logit_scale.exp().float() * (image_features @ text_features.T)
            logits = raw_logits / temperature
            probs = torch.softmax(logits, dim=-1) # Shape: (Batch, Num_Classes)
            
            # Storage for the dense, pixel-level heatmaps for this batch
            batch_dense_cams = torch.zeros((chunk.shape[0], num_classes, p_size, p_size), device=device)
            
            # --- THE GRAD-CAM BACKWARD LOOP ---
            for c_idx in range(num_classes):
                model.zero_grad()
                
                # Score for this specific class across the whole batch
                score = (image_features @ text_features[c_idx]) 
                
                # Sum the batch to allow simultaneous backward routing
                score.sum().backward(retain_graph=True)
                
                if hook.gradients is not None:
                    g, a = hook.gradients.clone().float(), hook.activations.clone().float()
                    w = torch.mean(g, dim=[2, 3], keepdim=True)
                    cam = F.relu(torch.sum(w * a, dim=1, keepdim=True)) # Shape: (Batch, 1, 7, 7)
                    
                    # Min-Max Normalization per patch
                    cam_min = cam.view(cam.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
                    cam_max = cam.view(cam.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
                    cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)
                    
                    # Upsample using nearest neighbor to preserve strict boundaries
                    cam_up = F.interpolate(cam_norm, size=(p_size, p_size), mode='nearest')
                    
                    # Modulate the spatial heatmap by the patch's overall softened probability
                    class_prob = probs[:, c_idx].view(-1, 1, 1, 1)
                    batch_dense_cams[:, c_idx, :, :] = (cam_up * class_prob).squeeze(1)
                
            # Move to CPU and format as (Batch, P_size, P_size, Num_Classes)
            all_dense_probs.extend(batch_dense_cams.permute(0, 2, 3, 1).detach().cpu().numpy())
            
            # Clear cache to prevent RTX 4070 OOM
            del chunk, image_features, raw_logits, logits, probs, batch_dense_cams
            torch.cuda.empty_cache()

        # 4. Native Block Projection
        for (u, l, left, right), dense_prob_block in zip(boxes, all_dense_probs):
            padded_scale_tensor[u:l, left:right, :] = dense_prob_block 

        # Slice off padding to get native dimensions
        local_scale_tensor = padded_scale_tensor[:H_orig, :W_orig, :]

        # 5. Mathematical Fusion (Tensor Accumulation with Matrix Weights)
        weighted_local_tensor = local_scale_tensor * current_weight_array
        fused_prob_tensor += weighted_local_tensor

    hook.close()

    # --- FINAL DECISION PHASE ---
    if fused_prob_tensor.max() > 0:
        fused_prob_tensor /= fused_prob_tensor.max()
    
    fused_class_map = np.argmax(fused_prob_tensor, axis=-1).astype(np.int32)
    fused_confidence_map = np.max(fused_prob_tensor, axis=-1)

    # =========================================================================
    # --- VISUALIZATION BLOCK ---
    # =========================================================================
    print("\nComposing Final Visual Map...")
    rgb_map_uint8 = segment_mask_to_rgb(fused_class_map, mapping_keys)
    rgb_map_float = rgb_map_uint8.astype(np.float32) / 255.0
    rgba_map = np.dstack((rgb_map_float, fused_confidence_map))

    original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    if sw_plot:
        fig, ax = plt.subplots(1, 2, figsize=(18, 9))
        ax[0].imshow(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB))
        ax[0].set_title(f"Original Image ({W_orig}x{H_orig})", fontsize=14)
        ax[0].axis('off')

        ax[1].imshow(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB))
        ax[1].imshow(rgba_map) 
        ax[1].set_title(f"BlackMagiCLIP Dense Matrix Fusion ({W_orig}x{H_orig})", fontsize=14)
        ax[1].axis('off')

        uavid_gt_colors = {
            "building": [128, 0, 0], "road": [128, 64, 128], "tree": [0, 128, 0],
            "low_veg": [128, 128, 0], "clutter": [0, 0, 0], "car": [192, 0, 192], "human": [64, 64, 0]
        }
        
        patches_legend = [
            mpatches.Patch(
                color=np.array(uavid_gt_colors.get(mapping_keys[i], [255, 255, 255])) / 255.0, 
                label=clip_prompts[i].split()[-1].capitalize()
            ) 
            for i in range(len(clip_prompts))
        ]
        ax[1].legend(handles=patches_legend, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

    return fused_class_map, fused_confidence_map

# =============================================================================
# --- 5. PIPELINE EXECUTION ---
# =============================================================================

if __name__ == "__main__":
    
    clip_prompts = [
        "aerial photo of a building", "aerial photo of road", "aerial photo of a tree",
        "aerial photo of low vegetation", "aerial photo of background clutter", 
        "aerial photo of car", "aerial photo of human"
    ]
    
    mapping_keys = ["building", "road", "tree", "low_veg", "clutter", "car", "human"]

    # --- THE SCALE-CLASS WEIGHT MATRIX ---
    scale_class_matrix = {
        448: { 
            "building": 1.2,
            "road": 0.95,
            "tree": 0.95,
            "low_veg": 2.8,
            "clutter": 1.5,
            "car": 0.0,
            "human": 0.0
        },
        224: { 
            "building": 1.0,
            "road": 0.9,
            "tree": 0.9,
            "low_veg": 2.5,
            "clutter": 1.5,
            "car": 1.1,
            "human": 0.8
        },
    }
    # Algorithm Hyperparameters
    fusion_temperature = 2.5 # Values > 1.0 soften the probability distribution

    # Define Image Directory
    image_dir = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_val\seq67\Images"
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Execute Pipeline
    sw_plot = True  # Set to False to skip visualization and process silently
    miou_lst = []
    per_class_lst = []
    for image_path in image_paths:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        fused_class_map, fused_conf_map = exact_multi_scale_ensemble_matrix(
            image_path, 
            clip_prompts, 
            mapping_keys, 
            scale_class_matrix, 
            temperature=fusion_temperature, # Injecting the parameter here
            sw_plot=sw_plot
        )

        results, miou = save_and_evaluate_single_image(
            image_path=image_path, 
            seg_map=fused_class_map, 
            mapping_keys=mapping_keys
    )
        miou_lst.append(miou)
        per_class_lst.append(results)
        sw_plot = False  # Only plot the first image for demonstration
        #input("Press Enter to continue to the next image...")
    print(f"\n=== FINAL AVERAGE mIoU across {len(miou_lst)} images: {np.mean(miou_lst):.4f} ===")
