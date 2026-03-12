import torch
import torch.nn.functional as F
import clip
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from tqdm import tqdm
import os
"""
Heuristic-Augmented Multi-Modal Explainable Segmentation (HAMMES)
This code implements an "Entropy-Alpha Composition" technique to visualize CLIP's patch-level predictions on UAVid images.
Furthermore Heurtistic-Weighted Panoptic Segmentation engine through Bayesian Prior injection to boost underperforming classes. 
"""

# --- 4070 CONTEXT INITIALIZATION ---
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.set_device(0)
    _ = torch.ones(1, device="cuda") @ torch.ones(1, device="cuda")

# --- SPATIAL HOOK ---
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

def multi_kernel_smoothing(seg_map, labels, kernel_dict):
    H, W = seg_map.shape
    final_smoothed = np.zeros((H, W), dtype=np.int32)
    
    # helper to find keywords in your full labels
    def get_k_size(full_label):
        for short_key, size in kernel_dict.items():
            if short_key in full_label.lower():
                return size
        return 3

    # Sort so that the largest 'Stuff' (Road/Buildings) are processed first
    sorted_labels = sorted(labels, key=lambda l: get_k_size(l), reverse=True)
    
    for label_name in sorted_labels:
        label_idx = labels.index(label_name)
        k_size = get_k_size(label_name)
        
        binary_mask = (seg_map == label_idx).astype(np.uint8)
        if binary_mask.sum() == 0: continue

        if k_size > 1:
            if k_size % 2 == 0: k_size += 1
            # Using medianBlur to kill the 'mini-patch' noise while keeping boundaries
            binary_mask = cv2.medianBlur(binary_mask, k_size)
        
        final_smoothed[binary_mask > 0] = label_idx
        
    return final_smoothed



def entropy_alpha_composition(image_path, labels, class_weights, patch_size=224, show_visualization=False, swSmoothing=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)
    model.eval()

    # UAVid Canonical Colors
    uavid_colors = np.array([
        [128, 0, 0], [128, 64, 128], [0, 128, 0], [128, 128, 0], [0, 0, 0], [192, 0, 192], [64, 64, 0]
    ])

    img_raw = Image.open(image_path).convert("RGB")
    W, H = img_raw.size
    cols, rows = W // patch_size, H // patch_size
    
    # OUTPUT MATRICES
    global_seg_map = np.zeros((H, W), dtype=np.int32)
    global_entropy_map = np.zeros((H, W), dtype=np.float32)

    # Pre-encode text
    text_tokens = clip.tokenize(labels).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    weights_vec = torch.tensor([class_weights.get(l, 1.0) for l in labels], device=device).float()
    hook = SpatialLayerHook(model.visual.layer4)

    print(f"Building Uncertainty-Aware Map: {rows*cols} Patches...")

    for r in range(rows):
        for c in tqdm(range(cols), desc=f"Row {r+1}"):
            left, upper = c * patch_size, r * patch_size
            patch = img_raw.crop((left, upper, left + patch_size, upper + patch_size))
            
            img_t = preprocess(patch).unsqueeze(0).to(device).type(model.dtype)
            img_t.requires_grad = True

            with torch.amp.autocast('cuda'):
                features_raw = model.encode_image(img_t)
                features = features_raw.clone().float()
                features = features / features.norm(dim=-1, keepdim=True)
                
                logits = model.logit_scale.exp().float() * (features @ text_features.T)
                weighted_logits = logits * weights_vec
                probs_t = torch.softmax(weighted_logits, dim=-1)
                probs = probs_t.detach().cpu().numpy()[0]

                # NORMALIZED ENTROPY: 0 (certain) to 1 (guessing)
                entropy = -torch.sum(probs_t * torch.log(probs_t + 1e-8)) / np.log(len(labels))
                entropy_val = entropy.item()

            # --- GRAD-CAM ---
            cams = []
            for i in range(len(labels)):
                model.zero_grad()
                score = (features @ text_features[i].unsqueeze(1))
                score.backward(retain_graph=True)
                torch.cuda.synchronize()
                
                if hook.gradients is not None:
                    g, a = hook.gradients.clone().float(), hook.activations.clone().float()
                    w = torch.mean(g, dim=[2, 3], keepdim=True)
                    cam = F.relu(torch.sum(w * a, dim=1, keepdim=True)).squeeze().cpu().detach().numpy()
                    cams.append(((cam - cam.min()) / (cam.max() + 1e-8)) * probs[i])
                else:
                    cams.append(np.zeros((7, 7)))

            # --- FUSION & STAMPING ---
            winning_map_small = np.argmax(np.stack(cams), axis=0)
            winning_map = cv2.resize(winning_map_small.astype(np.uint8), (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
            
            global_seg_map[upper:upper+patch_size, left:left+patch_size] = winning_map
            global_entropy_map[upper:upper+patch_size, left:left+patch_size] = entropy_val

            del img_t, features_raw, features
            torch.cuda.empty_cache()

    hook.close()

    # --- THE ENTROPY-ALPHA BLENDING LOGIC ---
    if swSmoothing:
        global_seg_map = multi_kernel_smoothing(global_seg_map, model_labels, custom_kernels)
    # 1. Convert segmentation map to an RGB image
    seg_rgb = uavid_colors[global_seg_map]
    
    # 2. Create the Alpha Channel: High Entropy = Low Alpha
    # We clip the entropy to ensure we don't go perfectly invisible (min alpha 0.1)
    # Alpha = (1 - Entropy) scaled to 0-255
    alpha_channel = (1.0 - global_entropy_map) * 255
    alpha_channel = np.clip(alpha_channel, 25, 255).astype(np.uint8)

    
    # 4. Merge into RGBA
    rgba_overlay = np.dstack((seg_rgb, alpha_channel)).astype(np.uint8)

    # --- VISUALIZATION ---
    if show_visualization:
        plt.figure(figsize=(20, 12))
        plt.imshow(np.array(img_raw))
        plt.imshow(rgba_overlay) # Overlays directly using the internal alpha channel
        
        plt.title("Entropy-Weighted Transparency Map (Faded = Uncertain)", fontsize=16)
        
        # Legend
        legend_handles = [mpatches.Patch(color=uavid_colors[i]/255.0, label=labels[i].split()[-1].capitalize()) for i in range(len(labels))]
        plt.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=6, fontsize=12)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    return global_seg_map

def segment_mask_to_rgb(global_seg_map, labels_order):
    """
    Converts a 2D segmentation map of integer indices into a 3D RGB image array.
    
    Args:
        global_seg_map (np.ndarray): 2D array of class indices (e.g., shape HxW).
        labels_order (list): A list of strings dictating which index maps to which class.
                             e.g., labels_order[0] = "building".
    Returns:
        np.ndarray: 3D array of shape (H, W, 3) in RGB format (uint8).
    """
    uavid_gt_colors = {
        "building":    [128, 0, 0],
        "road":        [128, 64, 128],
        "tree":        [0, 128, 0],
        "low_veg":     [128, 128, 0],
        "clutter":     [0, 0, 0],
        "car":         [192, 0, 192],
        "human":       [64, 64, 0]
    }

    # 1. Build the Color Matrix (N x 3)
    # This aligns the dictionary colors with the integer indices in your mask.
    num_classes = len(labels_order)
    color_matrix = np.zeros((num_classes, 3), dtype=np.uint8)
    
    for i, label in enumerate(labels_order):
        if label in uavid_gt_colors:
            color_matrix[i] = uavid_gt_colors[label]
        else:
            # Fallback color (e.g., white) if a label isn't in the dictionary
            color_matrix[i] = [255, 255, 255]

    # 2. Vectorized Matrix Mapping
    # NumPy replaces every integer 'i' in the 2D map with the 3-element array at color_matrix[i].
    rgb_image = color_matrix[global_seg_map]

    return rgb_image

# =============================================================================
# --- RUN EXECUTION BLOCK ---
# =============================================================================

# 1. The Prompts for CLIP (Maintains context for accuracy)
model_labels = [
    "aerial view of a building",       # 0
    "aerial view of road",             # 1
    "aerial view of a tree",           # 2
    "aerial view of low vegetation",   # 3
    "aerial view of background clutter",# 4
    "aerial view of car",              # 5
    "aerial view of human"             # 6
]

# 2. The Short Keys for the Logic/RGB Mapping Dictionaries
# CRITICAL: These MUST align perfectly with the indices of model_labels above
mapping_keys = [
    "building", "road", "tree", "low_veg", "clutter", "car", "human"
]

# 3. Physics-based Kernels (Using the short keys)
custom_kernels = {
    "building": 5*32,          
    "road": 5*32,              
    "tree": 3*32,              
    "low_veg": 4*32,
    "clutter": 2*32,            
    "car": 2*32, 
    "human": 1*32              
}

# 4. Bayesian Priors (Using the exact model_labels strings)
manual_weights = {
    "aerial view of road": 1.1, 
    "aerial view of background clutter": 1.1, 
    "aerial view of a building": 0.95
}

image_dir = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_val\seq16\Images"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    

# =============================================================================
# --- MODIFIED LOOP ---
# =============================================================================


for image_path in image_paths:
    # 1. Generate the map using the LONG prompts for CLIP
    global_seg_map = entropy_alpha_composition(image_path, model_labels, manual_weights)
    
    # 2. Convert to RGB using the SHORT mapping keys
    # This prevents the "White Output" bug by ensuring dictionary matches
    rgb_seg_map = segment_mask_to_rgb(global_seg_map, mapping_keys)
    
    # 3. OpenCV Color Space Conversion
    # CRITICAL: cv2.imwrite expects BGR, but segment_mask_to_rgb outputs RGB.
    # If we don't flip this, Red Cars become Blue Cars in the saved image.
    bgr_seg_map = cv2.cvtColor(rgb_seg_map, cv2.COLOR_RGB2BGR)
    
    # Optional: Display using OpenCV (Wait for a key press to continue to the next image)
    # cv2.imshow("RGB Segmentation Map", bgr_seg_map)
    # cv2.waitKey(0) 
    
    # 4. Save the Output
    # Create the target directory if it doesn't exist to prevent FileNotFoundError
    save_path = image_path.replace("Images", "Predictions")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cv2.imwrite(save_path, bgr_seg_map)
    print(f"Saved prediction to: {save_path}")

# cv2.destroyAllWindows()