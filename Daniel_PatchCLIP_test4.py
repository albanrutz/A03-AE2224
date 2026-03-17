import torch
import clip
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm
import os
from scoring_general import save_and_evaluate_single_image

# --- 4070 CONTEXT INITIALIZATION ---
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.set_device(0)
    _ = torch.ones(1, device="cuda") @ torch.ones(1, device="cuda")

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

def exact_multi_scale_ensemble_matrix(image_path, clip_prompts, mapping_keys, scale_class_matrix, sw_plot=True):
    """
    Multi-Scale ensemble that applies a specific weight to each class at each scale,
    with built-in Matplotlib visualization.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 1. Global Setup
    original_image = Image.open(image_path).convert("RGB")
    W_orig, H_orig = original_image.size
    num_classes = len(clip_prompts)
    
    # 3D Probability Tensor: (Height, Width, Num_Classes)
    fused_prob_tensor = np.zeros((H_orig, W_orig, num_classes), dtype=np.float32)

    text_tokens = clip.tokenize(clip_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Extract the patch scales from the matrix keys and sort descending
    patch_scales = sorted(list(scale_class_matrix.keys()), reverse=True)

    # 2. Iterate through each Spatial Scale
    for p_size in patch_scales:
        print(f"\n--- Processing Scale: {p_size}x{p_size} ---")
        
        # Extract the specific class weights for THIS scale
        scale_weights_dict = scale_class_matrix[p_size]
        
        # Convert dictionary to a 1D NumPy array strictly aligned with mapping_keys
        current_weight_array = np.array(
            [scale_weights_dict.get(k, 1.0) for k in mapping_keys], 
            dtype=np.float32
        )
        
        pad_w = (p_size - (W_orig % p_size)) % p_size
        pad_h = (p_size - (H_orig % p_size)) % p_size
        padded_w, padded_h = W_orig + pad_w, H_orig + pad_h
        
        padded_img = Image.new("RGB", (padded_w, padded_h), color=(0, 0, 0))
        padded_img.paste(original_image, (0, 0))
        
        cols, rows = padded_w // p_size, padded_h // p_size
        padded_scale_tensor = np.zeros((padded_h, padded_w, num_classes), dtype=np.float32)

        patches = []
        boxes = [] 
        
        for r in range(rows):
            for c in range(cols):
                left, upper = c * p_size, r * p_size
                right, lower = left + p_size, upper + p_size
                
                patch = padded_img.crop((left, upper, right, lower))
                patches.append(preprocess(patch))
                boxes.append((upper, lower, left, right))

        batch_tensor = torch.stack(patches).to(device)
        batch_size = 128 
        
        all_probs = []
        with torch.no_grad():
            logit_scale = model.logit_scale.exp()
            for i in tqdm(range(0, len(batch_tensor), batch_size), desc=f"Evaluating Patches"):
                chunk = batch_tensor[i : i + batch_size]
                
                with torch.amp.autocast('cuda'):
                    image_features = model.encode_image(chunk)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    logits = logit_scale * (image_features @ text_features.T)
                    
                    # Compute raw Softmax probabilities
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    all_probs.extend(probs)

        # Project probabilities into the padded tensor
        for (u, l, left, right), prob_dist in zip(boxes, all_probs):
            padded_scale_tensor[u:l, left:right, :] = prob_dist 

        # Slice off padding to get native dimensions
        local_scale_tensor = padded_scale_tensor[:H_orig, :W_orig, :]

        # 3. MATHEMATICAL FUSION (The Matrix Multiplication)
        weighted_local_tensor = local_scale_tensor * current_weight_array
        fused_prob_tensor += weighted_local_tensor

    # 4. Final Decision Phase
    if fused_prob_tensor.max() > 0:
        fused_prob_tensor /= fused_prob_tensor.max()
    
    fused_class_map = np.argmax(fused_prob_tensor, axis=-1).astype(np.int32)
    fused_confidence_map = np.max(fused_prob_tensor, axis=-1)

    # =========================================================================
    # 5. MATPLOTLIB VISUALIZATION BLOCK
    # =========================================================================
    print("\nComposing Final Visual Map...")
    
    # Generate the RGB Array based on the winning classes
    rgb_map_uint8 = segment_mask_to_rgb(fused_class_map, mapping_keys)
    rgb_map_float = rgb_map_uint8.astype(np.float32) / 255.0
    
    # Build the RGBA Alpha-Blended overlay using the confidence map
    rgba_map = np.dstack((rgb_map_float, fused_confidence_map))

    original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    if sw_plot:
        fig, ax = plt.subplots(1, 2, figsize=(18, 9))
        
        ax[0].imshow(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB))
        ax[0].set_title(f"Original Image ({W_orig}x{H_orig})", fontsize=14)
        ax[0].axis('off')

        # Layer the original image behind the translucent probability map
        ax[1].imshow(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB))
        ax[1].imshow(rgba_map) 
        ax[1].set_title(f"Matrix-Weighted Soft Voting Fusion ({W_orig}x{H_orig})", fontsize=14)
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
# --- EXECUTION ---
# =============================================================================

clip_prompts = [
    "aerial view of a building", "aerial view of road", "aerial view of a tree",
    "aerial view of low vegetation", "aerial view of background clutter", 
    "aerial view of car", "aerial view of human"
]

mapping_keys = ["building", "road", "tree", "low_veg", "clutter", "car", "human"]

# --- THE SCALE-CLASS WEIGHT MATRIX ---
# Format: { Patch_Size: { "class_key": weight_multiplier } }

scale_class_matrix = {
    448: { # Massive Context: Trust it for geography, forbid it from guessing objects.
        "building": 1.0, "road": 1.3, "tree": 1.0, "low_veg": 1.0, 
        "clutter": 1.0, "car": 0.0, "human": 0.0 
    },
    224: { # Medium Context
        "building": 1.0, "road": 1.3, "tree": 1.0, "low_veg": 1.0, 
        "clutter": 1.0, "car": 0.0, "human": 0.0
    },
    112: { # Fine Context
        "building": 1.0, "road": 1.2, "tree": 1.0, "low_veg": 1.0, 
        "clutter": 1.0, "car": 0.2, "human": 0.0
    },
    56: { # Micro Context: Highly suppress geography to let small objects punch through.
        "building": 0.5, "road": 1.0, "tree": 1.0, "low_veg": 1.0, 
        "clutter": 0.5, "car": 0.8, "human": 0.2,
    },
}

image_dir = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_val\seq16\Images"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    

# Execute
sw_plot = True  # Set to False to skip visualization and just save predictions
for image_path in image_paths:
    fused_class_map, fused_conf_map = exact_multi_scale_ensemble_matrix(
        image_path, clip_prompts, mapping_keys, scale_class_matrix, sw_plot=sw_plot
    )
    results, miou = save_and_evaluate_single_image(
    image_path=image_path, 
    seg_map=fused_class_map, 
    mapping_keys=mapping_keys
)