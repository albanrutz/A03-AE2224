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
import time 
import pandas as pd

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

def exact_multi_scale_ensemble_matrix(image_path, clip_prompts, mapping_keys, scale_class_matrix, scale_threshold_matrix=None, temperature=1.0, sw_plot=True):
    """
    Multi-Scale ensemble that applies a specific weight to each class at each scale,
    with a Hard Activation Gate to mathematically crush low-confidence hallucinations.
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
    
    if scale_threshold_matrix is None:
        scale_threshold_matrix = {}

    # 2. Iterate through each Spatial Scale
    for p_size in patch_scales:
        print(f"\n--- Processing Scale: {p_size}x{p_size} ---")
        
        # Extract the specific class weights for THIS scale
        scale_weights_dict = scale_class_matrix[p_size]
        current_weight_array = np.array(
            [scale_weights_dict.get(k, 1.0) for k in mapping_keys], 
            dtype=np.float32
        )
        
        # Extract specific class hard-thresholds for THIS scale
        scale_thresholds_dict = scale_threshold_matrix.get(p_size, {})
        current_threshold_array = torch.tensor(
            [scale_thresholds_dict.get(k, 0.0) for k in mapping_keys], 
            device=device, dtype=torch.float32
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
                    
                    # Compute Raw Logits
                    raw_logits = logit_scale * (image_features @ text_features.T)
                    
                    # 1. RAW Probability (Strict confidence for evaluating thresholds)
                    raw_probs = torch.softmax(raw_logits, dim=-1)
                    
                    # 2. SOFT Probability (If temperature is used, otherwise identical)
                    logits = raw_logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    
                    # --- THE HARD ACTIVATION GATE ---
                    # Crushes the probability to 0.0 if it doesn't meet the specified minimum bound
                    probs = torch.where(raw_probs < current_threshold_array, torch.zeros_like(probs), probs)
                    
                    all_probs.extend(probs.cpu().numpy())

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
    if sw_plot:
        print("\nComposing Final Visual Map...")
        rgb_map_uint8 = segment_mask_to_rgb(fused_class_map, mapping_keys)
        rgb_map_float = rgb_map_uint8.astype(np.float32) / 255.0
        rgba_map = np.dstack((rgb_map_float, fused_confidence_map))

        original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        fig, ax = plt.subplots(1, 2, figsize=(18, 9))
        
        ax[0].imshow(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB))
        ax[0].set_title(f"Original Image ({W_orig}x{H_orig})", fontsize=14)
        ax[0].axis('off')

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
    "drone photo of a building", "drone photo of a road", "drone photo of a tree",
    "drone photo of low vegetation", "drone photo of background clutter", 
    "drone photo of a car", "drone photo of a human"
]

mapping_keys = ["building", "road", "tree", "low_veg", "clutter", "car", "human"]

# --- THE SCALE-CLASS WEIGHT MATRIX ---
# Format: { Patch_Size: { "class_key": weight_multiplier } }

scale_class_matrix = {
    448: { 
        "building": 0.8, "road": 0.8, "tree": 0.8, "low_veg": 0.8, 
        "clutter": 0.8, "car": 0.0, "human": 0.0 
    },
    224: { 
        "building": 1.0, "road": 1.0, "tree": 1.0, "low_veg": 1.0, 
        "clutter": 1.5, "car": 0.0, "human": 0.0
    },
    112: { 
        "building": 1.2, "road": 1.2, "tree": 1.1, "low_veg": 1.1, 
        "clutter": 1.7, "car": 1.4, "human": 0.0
    },
    56: { 
        "building": 0.5, "road": 1.0, "tree": 1.0, "low_veg": 1.0, 
        "clutter": 1.2, "car": 1.6, "human": 5.0
    },
}

# --- THE HARD ACTIVATION GATE MATRIX ---
# Format: { Patch_Size: { "class_key": minimum_probability_0_to_1 } }
scale_threshold_matrix = {
    448: { 
        "building": 0.0, "road": 0.0, "tree": 0.0, "low_veg": 0.0, 
        "clutter": 0.0, "car": 0.0, "human": 0.0 
    },
    224: { 
        "building": 0.0, "road": 0.0, "tree": 0.0, "low_veg": 0.0, 
        "clutter": 0.0, "car": 0.0, "human": 0.0
    },
    112: { 
        "building": 0.0, "road": 0.0, "tree": 0.0, "low_veg": 0.0, 
        "clutter": 0.0, "car": 0.7, "human": 0.0
    },
    56: { 
        "building": 0.0, "road": 0.0, "tree": 0.0, "low_veg": 0.0, 
        "clutter": 0.0, "car": 0.7, "human": 0.85
    },
}
test_dir = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1\Images"
cv_dir = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_val\seq67\Images"

sw_plot = False  # Set to False to skip visualization and just save predictions
image_dir = cv_dir
#image_dir = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_val\seq16\Images"
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
miou_lst = []
weighted_miou_lst = []
per_class_lst = []
time_lst = []

# Execute

for image_path in image_paths:
    start_time = time.time()

    fused_class_map, fused_conf_map = exact_multi_scale_ensemble_matrix(
        image_path,
        clip_prompts,
        mapping_keys,
        scale_class_matrix,
        scale_threshold_matrix=scale_threshold_matrix,
        temperature=1.0,
        sw_plot=sw_plot
    )

    results, miou, weighted_miou = save_and_evaluate_single_image(
        image_path=image_path,
        seg_map=fused_class_map,
        mapping_keys=mapping_keys
    )

    if results is not None:
        miou_lst.append(miou)
        weighted_miou_lst.append(weighted_miou)
        per_class_lst.append(results)
    time_lst.append(time.time() - start_time)

if miou_lst:
    print(f"\n=== FINAL AVERAGE mIoU across {len(miou_lst)} images: {np.mean(miou_lst):.4f} ===")
    print(f"=== FINAL AVERAGE Weighted mIoU across {len(weighted_miou_lst)} images: {np.mean(weighted_miou_lst):.4f} ===")
    print(f"=== FINAL AVERAGE Inference Time across {len(time_lst)} images: {np.mean(time_lst):.4f} seconds ===")

    # per_class_lst is a list of {class_name: {metric: value}} dicts.
    # Convert each to a DataFrame with classes as rows, metrics as columns,
    # then average across images.
    dataframes = [pd.DataFrame(run).T for run in per_class_lst]  # .T → rows=classes, cols=metrics
    avg_df = sum(dataframes) / len(dataframes)

    header = f"\n{'='*84}"
    header += f"\nImage: Average Results"
    header += f"\n{'='*84}"
    print(header)

    col_w = 20
    print(f"\n{'Category':<{col_w}} {'IoU':>8} {'F0.5':>8} {'F1':>8} {'F2':>8} "
          f"{'Precision':>10} {'Recall':>8}")
    print("-" * 76)

    for name, m in avg_df.iterrows():  # iterrows() since rows=classes now
        print(f"{name:<{col_w}} {m['IoU']:>8.4f} {m['F0.5']:>8.4f} {m['F1']:>8.4f} "
              f"{m['F2']:>8.4f} {m['Precision']:>10.4f} {m['Recall']:>8.4f}")

    print("-" * 76)
    avg_miou = avg_df['IoU'].mean()
    avg_weighted_miou = np.mean(weighted_miou_lst)
    print(f"\nmIoU:          {avg_miou:.4f}  ({avg_miou*100:.2f}%)")
    print(f"Weighted mIoU: {avg_weighted_miou:.4f}  ({avg_weighted_miou*100:.2f}%)\n")