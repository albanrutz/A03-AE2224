import torch
import torch.nn as nn
import clip
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm
import os
from releases.support.scoring_general import save_and_evaluate_single_image
import time
import pandas as pd

# --- 4070 CONTEXT INITIALIZATION ---
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.set_device(0)
    _ = torch.ones(1, device="cuda") @ torch.ones(1, device="cuda")

# =============================================================================
# LEARNABLE WEIGHT MATRICES
# =============================================================================

class ScaleWeights(nn.Module):
    """
    Holds all optimizable parameters for the multi-scale ensemble:

      scale_class_weights  – raw (unconstrained) weights; positive values
                             enforced via softplus during forward pass so that
                             the fused probability tensor is always non-negative.

      scale_threshold_raw  – unconstrained parameters mapped through sigmoid
                             to [0, 1] so the hard-activation gate threshold
                             is always a valid probability.
    """

    def __init__(self, patch_scales, mapping_keys,
                 init_class_matrix, init_threshold_matrix):
        super().__init__()
        self.patch_scales = patch_scales          # list[int], sorted descending
        self.mapping_keys = mapping_keys          # list[str]
        num_classes = len(mapping_keys)
        key_to_idx = {k: i for i, k in enumerate(mapping_keys)}

        # ---- scale_class_weights ------------------------------------------------
        # One learnable scalar per (scale, class) pair, stored as a single
        # (num_scales × num_classes) parameter matrix for clean gradient flow.
        class_init = torch.zeros(len(patch_scales), num_classes)
        for s_idx, p in enumerate(patch_scales):
            d = init_class_matrix.get(p, {})
            for k, v in d.items():
                if k in key_to_idx:
                    # Inverse softplus so that softplus(x) ≈ v at init
                    # softplus(x) = log(1 + exp(x))  →  x = log(exp(v) - 1)
                    v = max(v, 1e-4)                      # guard log(0)
                    class_init[s_idx, key_to_idx[k]] = float(
                        torch.log(torch.tensor(v).exp() - 1 + 1e-6)
                    )
        self.scale_class_raw = nn.Parameter(class_init)

        # ---- scale_threshold_raw ------------------------------------------------
        # Sigmoid maps ℝ → (0, 1); we initialise via inverse-sigmoid so that
        # sigmoid(x) ≈ threshold at the start of training.
        thresh_init = torch.zeros(len(patch_scales), num_classes)
        for s_idx, p in enumerate(patch_scales):
            d = init_threshold_matrix.get(p, {})
            for k, v in d.items():
                if k in key_to_idx:
                    v = float(np.clip(v, 1e-4, 1 - 1e-4))
                    thresh_init[s_idx, key_to_idx[k]] = float(
                        np.log(v / (1.0 - v))             # logit / inverse-sigmoid
                    )
        self.scale_threshold_raw = nn.Parameter(thresh_init)

    # ------------------------------------------------------------------
    def class_weights(self):
        """Returns softplus-activated weights: shape (num_scales, num_classes)."""
        return torch.nn.functional.softplus(self.scale_class_raw)

    def thresholds(self):
        """Returns sigmoid-activated thresholds in [0, 1]: shape (num_scales, num_classes)."""
        return torch.sigmoid(self.scale_threshold_raw)

    # ------------------------------------------------------------------
    def as_dicts(self):
        """Export current parameters back to the dict format used by the pipeline."""
        cw = self.class_weights().detach().cpu().numpy()
        th = self.thresholds().detach().cpu().numpy()

        class_matrix   = {}
        threshold_matrix = {}
        for s_idx, p in enumerate(self.patch_scales):
            class_matrix[p]    = {k: float(cw[s_idx, i]) for i, k in enumerate(self.mapping_keys)}
            threshold_matrix[p] = {k: float(th[s_idx, i]) for i, k in enumerate(self.mapping_keys)}
        return class_matrix, threshold_matrix


# =============================================================================
# GT LABEL LOADING
# =============================================================================

# UAVid RGB colour → class index lookup
_UAVID_COLOR_TO_IDX = {
    (128,   0,   0): 0,   # building
    (128,  64, 128): 1,   # road
    (  0, 128,   0): 2,   # tree
    (128, 128,   0): 3,   # low_veg
    (  0,   0,   0): 4,   # clutter
    (192,   0, 192): 5,   # car
    ( 64,  64,   0): 6,   # human
}

def load_gt_label(image_path: str, mapping_keys: list) -> np.ndarray | None:
    """
    Loads the UAVid GT label mask corresponding to `image_path`.

    The label image lives at the same path with "Images" replaced by "Labels".
    UAVid labels are RGB PNGs where each pixel colour encodes a class.

    Returns
    -------
    label_map : np.ndarray of shape (H, W), dtype int64, values in [0, C-1]
                or None if no label file is found.
    """
    label_path = image_path.replace(os.sep + "Images" + os.sep,
                                    os.sep + "Labels" + os.sep)
    # Also handle forward-slash paths (Linux / mixed)
    label_path = label_path.replace("/Images/", "/Labels/")

    if not os.path.exists(label_path):
        print(f"  [WARNING] No label found for: {image_path}")
        return None

    label_rgb = np.array(Image.open(label_path).convert("RGB"), dtype=np.uint8)
    H, W, _ = label_rgb.shape
    label_map = np.full((H, W), fill_value=mapping_keys.index("clutter"),
                        dtype=np.int64)   # default → clutter (index 4)

    for (r, g, b), cls_idx in _UAVID_COLOR_TO_IDX.items():
        mask = (label_rgb[:, :, 0] == r) & \
               (label_rgb[:, :, 1] == g) & \
               (label_rgb[:, :, 2] == b)
        label_map[mask] = cls_idx

    return label_map


# =============================================================================
# CORE SEGMENTATION FUNCTION  (unchanged interface)
# =============================================================================

def segment_mask_to_rgb(global_seg_map, labels_order):
    uavid_gt_colors = {
        "building": [128, 0, 0],  "road":    [128, 64, 128],
        "tree":     [0, 128, 0],  "low_veg": [128, 128, 0],
        "clutter":  [0, 0, 0],    "car":     [192, 0, 192],
        "human":    [64, 64, 0]
    }
    num_classes = len(labels_order)
    color_matrix = np.zeros((num_classes, 3), dtype=np.uint8)
    for i, label in enumerate(labels_order):
        color_matrix[i] = uavid_gt_colors.get(label, [255, 255, 255])
    return color_matrix[global_seg_map]


def exact_multi_scale_ensemble_matrix(image_path, clip_prompts, mapping_keys,
                                       scale_class_matrix, scale_threshold_matrix=None,
                                       temperature=1.0, sw_plot=True):
    """
    Multi-Scale ensemble with Hard Activation Gate.
    Identical interface to the original; uses the dict form of the weight matrices.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    original_image = Image.open(image_path).convert("RGB")
    W_orig, H_orig = original_image.size
    num_classes = len(clip_prompts)

    fused_prob_tensor = np.zeros((H_orig, W_orig, num_classes), dtype=np.float32)

    text_tokens = clip.tokenize(clip_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    patch_scales = sorted(list(scale_class_matrix.keys()), reverse=True)

    if scale_threshold_matrix is None:
        scale_threshold_matrix = {}

    for p_size in patch_scales:
        print(f"\n--- Processing Scale: {p_size}x{p_size} ---")

        scale_weights_dict    = scale_class_matrix[p_size]
        current_weight_array  = np.array(
            [scale_weights_dict.get(k, 1.0) for k in mapping_keys], dtype=np.float32)

        scale_thresholds_dict  = scale_threshold_matrix.get(p_size, {})
        current_threshold_array = torch.tensor(
            [scale_thresholds_dict.get(k, 0.0) for k in mapping_keys],
            device=device, dtype=torch.float32)

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
                patches.append(preprocess(padded_img.crop((left, upper, right, lower))))
                boxes.append((upper, lower, left, right))

        batch_tensor = torch.stack(patches).to(device)
        batch_size = 128
        all_probs = []

        with torch.no_grad():
            logit_scale = model.logit_scale.exp()
            for i in tqdm(range(0, len(batch_tensor), batch_size), desc="Evaluating Patches"):
                chunk = batch_tensor[i: i + batch_size]
                with torch.amp.autocast('cuda'):
                    image_features = model.encode_image(chunk)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    raw_logits = logit_scale * (image_features @ text_features.T)
                    raw_probs  = torch.softmax(raw_logits, dim=-1)
                    logits     = raw_logits / temperature
                    probs      = torch.softmax(logits, dim=-1)
                    probs      = torch.where(raw_probs < current_threshold_array,
                                             torch.zeros_like(probs), probs)
                    all_probs.extend(probs.cpu().numpy())

        for (u, l, left, right), prob_dist in zip(boxes, all_probs):
            padded_scale_tensor[u:l, left:right, :] = prob_dist

        local_scale_tensor = padded_scale_tensor[:H_orig, :W_orig, :]
        fused_prob_tensor += local_scale_tensor * current_weight_array

    if fused_prob_tensor.max() > 0:
        fused_prob_tensor /= fused_prob_tensor.max()

    fused_class_map      = np.argmax(fused_prob_tensor, axis=-1).astype(np.int32)
    fused_confidence_map = np.max(fused_prob_tensor, axis=-1)

    if sw_plot:
        print("\nComposing Final Visual Map...")
        rgb_map_uint8 = segment_mask_to_rgb(fused_class_map, mapping_keys)
        rgb_map_float = rgb_map_uint8.astype(np.float32) / 255.0
        rgba_map      = np.dstack((rgb_map_float, fused_confidence_map))
        original_cv   = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

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
            "low_veg":  [128, 128, 0], "clutter": [0, 0, 0],
            "car":      [192, 0, 192], "human": [64, 64, 0]
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

    return fused_class_map, fused_confidence_map, fused_prob_tensor   # <-- also return raw tensor


# =============================================================================
# OPTIMISATION LOOP
# =============================================================================

def optimise_weights(image_paths, clip_prompts, mapping_keys,
                     init_class_matrix, init_threshold_matrix,
                     num_epochs=10, lr=1e-2, temperature=1.0):
    """
    Run Adam optimisation over scale_class_matrix and scale_threshold_matrix.

    The loss is a differentiable surrogate computed on the fused probability
    tensor (see `surrogate_loss`).  Because the CLIP forward pass is frozen and
    non-differentiable w.r.t. the weight matrices (they are applied *after* the
    CLIP inference), we:
      1. Run the CLIP patch-scoring once per image to get per-scale prob maps.
      2. Cache those maps.
      3. Re-apply the learnable weights inside an autograd-enabled loop so
         gradients flow back to `ScaleWeights`.

    Parameters
    ----------
    image_paths      : list[str]   – training images
    num_epochs       : int         – optimisation epochs
    lr               : float       – Adam learning rate
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch_scales = sorted(list(init_class_matrix.keys()), reverse=True)

    # ---- Initialise learnable parameters ------------------------------------
    sw = ScaleWeights(patch_scales, mapping_keys,
                      init_class_matrix, init_threshold_matrix).to(device)
    optimizer = torch.optim.Adam(sw.parameters(), lr=lr)

    # ---- Pre-compute CLIP patch scores (expensive; done once) ---------------
    print("\n" + "="*60)
    print("  Pre-computing CLIP patch scores (cached for optimisation)")
    print("="*60)

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    text_tokens = clip.tokenize(clip_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    num_classes = len(clip_prompts)

    # cache[image_path][p_size] = (H_orig, W_orig, raw_prob_map np float32 H×W×C)
    cache = {}

    for image_path in tqdm(image_paths, desc="Caching images"):
        original_image = Image.open(image_path).convert("RGB")
        W_orig, H_orig = original_image.size
        cache[image_path] = {"size": (H_orig, W_orig), "scales": {}}

        for p_size in patch_scales:
            pad_w = (p_size - (W_orig % p_size)) % p_size
            pad_h = (p_size - (H_orig % p_size)) % p_size
            padded_w, padded_h = W_orig + pad_w, H_orig + pad_h

            padded_img = Image.new("RGB", (padded_w, padded_h), color=(0, 0, 0))
            padded_img.paste(original_image, (0, 0))

            cols, rows = padded_w // p_size, padded_h // p_size
            patches, boxes = [], []

            for r in range(rows):
                for c in range(cols):
                    left, upper = c * p_size, r * p_size
                    right, lower = left + p_size, upper + p_size
                    patches.append(preprocess(padded_img.crop((left, upper, right, lower))))
                    boxes.append((upper, lower, left, right))

            batch_tensor = torch.stack(patches).to(device)
            all_probs = []

            with torch.no_grad():
                logit_scale = model.logit_scale.exp()
                for i in range(0, len(batch_tensor), 128):
                    chunk = batch_tensor[i: i + 128]
                    with torch.amp.autocast('cuda'):
                        img_feat = model.encode_image(chunk)
                        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                        raw_logits = logit_scale * (img_feat @ text_features.T)
                        raw_probs  = torch.softmax(raw_logits, dim=-1)
                        logits     = raw_logits / temperature
                        probs      = torch.softmax(logits, dim=-1)
                        all_probs.extend(probs.cpu().numpy())

            # Store the *unweighted*, *unthresholded* probs for this scale
            # (thresholds will be re-applied during the optimisation loop)
            raw_prob_map = np.zeros((padded_h, padded_w, num_classes), dtype=np.float32)
            # Also store raw_probs per patch for threshold gating
            raw_prob_map_raw = np.zeros((padded_h, padded_w, num_classes), dtype=np.float32)

            for idx, ((u, l, left, right), prob_dist) in enumerate(zip(boxes, all_probs)):
                raw_prob_map[u:l, left:right, :] = prob_dist

            # We need raw_probs too to apply the hard-gate during opt
            all_raw_probs = []
            with torch.no_grad():
                logit_scale = model.logit_scale.exp()
                for i in range(0, len(batch_tensor), 128):
                    chunk = batch_tensor[i: i + 128]
                    with torch.amp.autocast('cuda'):
                        img_feat = model.encode_image(chunk)
                        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                        raw_logits = logit_scale * (img_feat @ text_features.T)
                        raw_p = torch.softmax(raw_logits, dim=-1)
                        all_raw_probs.extend(raw_p.cpu().numpy())

            for (u, l, left, right), rp in zip(boxes, all_raw_probs):
                raw_prob_map_raw[u:l, left:right, :] = rp

            cache[image_path]["scales"][p_size] = {
                "softmax_probs": raw_prob_map[:H_orig, :W_orig, :],      # H×W×C
                "raw_probs":     raw_prob_map_raw[:H_orig, :W_orig, :]   # H×W×C
            }

        # Load GT label for this image (RGB PNG → class index map)
        gt = load_gt_label(image_path, mapping_keys)
        cache[image_path]["gt"] = gt   # np.ndarray (H, W) int64, or None

    # ---- Compute inverse-frequency class weights from GT --------------------
    # Pixels from all training images are pooled to estimate class frequencies.
    print("\nComputing inverse-frequency class weights from GT masks...")
    class_counts = np.zeros(num_classes, dtype=np.float64)
    for image_path in image_paths:
        gt = cache[image_path]["gt"]
        if gt is not None:
            for c in range(num_classes):
                class_counts[c] += (gt == c).sum()

    total_pixels = class_counts.sum()
    # Inverse frequency: rare classes get higher weight, capped to avoid extreme values
    class_freq = class_counts / (total_pixels + 1e-8)
    inv_freq = 1.0 / (class_freq + 1e-6)
    inv_freq = inv_freq / inv_freq.mean()          # normalise so mean weight = 1
    inv_freq = np.clip(inv_freq, 0.1, 10.0)        # cap to [0.1, 10] for stability
    class_loss_weights = torch.tensor(inv_freq, dtype=torch.float32, device=device)

    print("  Per-class loss weights (inv-freq, normalised):")
    for i, k in enumerate(mapping_keys):
        print(f"    {k:<12}: {class_loss_weights[i].item():.3f}  "
              f"(freq={class_freq[i]*100:.2f}%)")

    # ---- Optimisation epochs ------------------------------------------------
    print("\n" + "="*60)
    print("  Starting Adam optimisation  [GT cross-entropy loss]")
    print("="*60)

    loss_history = []
    ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_loss_weights)

    # Spatial subsampling stride: every 4th pixel → ~6% of pixels, still dense enough
    STRIDE = 4

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_images_with_gt = 0
        optimizer.zero_grad()

        cw  = sw.class_weights()   # (num_scales, num_classes) softplus, grad-enabled
        thr = sw.thresholds()      # (num_scales, num_classes) sigmoid,  grad-enabled

        for image_path in image_paths:
            gt = cache[image_path]["gt"]
            if gt is None:
                continue   # skip images without a label

            H_orig, W_orig = cache[image_path]["size"]

            # Build the fused tensor in torch so gradients flow to cw / thr
            fused = torch.zeros(H_orig, W_orig, num_classes,
                                dtype=torch.float32, device=device)

            for s_idx, p_size in enumerate(patch_scales):
                sp_t = torch.tensor(
                    cache[image_path]["scales"][p_size]["softmax_probs"],
                    dtype=torch.float32, device=device)          # H×W×C
                rp_t = torch.tensor(
                    cache[image_path]["scales"][p_size]["raw_probs"],
                    dtype=torch.float32, device=device)          # H×W×C

                # Hard-activation gate with learned thresholds
                # Using sigmoid-smooth gate (straight-through for gradient) via
                # a steep sigmoid instead of hard step so gradients aren't zero
                gate = torch.sigmoid(100.0 * (rp_t - thr[s_idx]))  # H×W×C ≈ step
                gated   = sp_t * gate
                weighted = gated * cw[s_idx]
                fused    = fused + weighted

            # Normalise
            fused_max = fused.max()
            if fused_max > 0:
                fused = fused / fused_max

            # Subsample spatially for memory efficiency
            fused_sub = fused[::STRIDE, ::STRIDE, :]          # (H', W', C)
            gt_sub    = gt[::STRIDE, ::STRIDE]                # (H', W')

            # CrossEntropyLoss expects (N, C) logits and (N,) targets
            H_, W_ = fused_sub.shape[:2]
            logits  = fused_sub.reshape(-1, num_classes)       # (N, C)
            targets = torch.tensor(gt_sub.reshape(-1),
                                   dtype=torch.long, device=device)  # (N,)

            # GT cross-entropy (the real, grounded loss)
            loss_ce = ce_loss_fn(logits, targets)

            # Tiny L2 to prevent unbounded weight growth (much smaller than before)
            loss_l2 = 1e-4 * (cw ** 2).sum()

            image_loss = loss_ce + loss_l2
            image_loss.backward(retain_graph=(image_path != image_paths[-1]))
            epoch_loss += image_loss.item()
            n_images_with_gt += 1

        optimizer.step()

        avg_loss = epoch_loss / max(n_images_with_gt, 1)
        loss_history.append(avg_loss)

        # Print current weights
        cw_np  = sw.class_weights().detach().cpu().numpy()
        thr_np = sw.thresholds().detach().cpu().numpy()

        print(f"\nEpoch {epoch+1}/{num_epochs}  |  Avg CE Loss: {avg_loss:.4f}  "
              f"(over {n_images_with_gt} images)")
        print(f"  Class weights (softplus):")
        for s_idx, p in enumerate(patch_scales):
            row = {k: f"{cw_np[s_idx, i]:.3f}" for i, k in enumerate(mapping_keys)}
            print(f"    p={p}: {row}")
        print(f"  Thresholds (sigmoid → [0,1]):")
        for s_idx, p in enumerate(patch_scales):
            row = {k: f"{thr_np[s_idx, i]:.3f}" for i, k in enumerate(mapping_keys)}
            print(f"    p={p}: {row}")

    print("\n" + "="*60)
    print("  Optimisation complete.")
    print("="*60)

    # Return optimised dicts (drop-in replacement for the originals)
    opt_class_matrix, opt_threshold_matrix = sw.as_dicts()
    return opt_class_matrix, opt_threshold_matrix, loss_history


# =============================================================================
# CONFIGURATION
# =============================================================================

clip_prompts = [
    "drone photo of a building", "drone photo of a road", "drone photo of a tree",
    "drone photo of low vegetation", "drone photo of background clutter",
    "drone photo of a car", "drone photo of a human"
]

mapping_keys = ["building", "road", "tree", "low_veg", "clutter", "car", "human"]

scale_class_matrix = {
    448: {"building": 0.8,  "road": 0.8,  "tree": 0.8,  "low_veg": 0.8,
          "clutter": 0.8,   "car": 0.0,   "human": 0.0},
    224: {"building": 1.0,  "road": 1.0,  "tree": 1.0,  "low_veg": 1.0,
          "clutter": 1.5,   "car": 0.0,   "human": 0.0},
    112: {"building": 1.2,  "road": 1.2,  "tree": 1.1,  "low_veg": 1.1,
          "clutter": 1.7,   "car": 1.4,   "human": 0.0},
    56:  {"building": 0.5,  "road": 1.0,  "tree": 1.0,  "low_veg": 1.0,
          "clutter": 1.2,   "car": 1.6,   "human": 5.0},
}

scale_threshold_matrix = {
    448: {"building": 0.0, "road": 0.0, "tree": 0.0, "low_veg": 0.0,
          "clutter": 0.0,  "car": 0.0,  "human": 0.0},
    224: {"building": 0.0, "road": 0.0, "tree": 0.0, "low_veg": 0.0,
          "clutter": 0.0,  "car": 0.0,  "human": 0.0},
    112: {"building": 0.0, "road": 0.0, "tree": 0.0, "low_veg": 0.0,
          "clutter": 0.0,  "car": 0.7,  "human": 0.0},
    56:  {"building": 0.0, "road": 0.0, "tree": 0.0, "low_veg": 0.0,
          "clutter": 0.0,  "car": 0.7,  "human": 0.85},
}

# =============================================================================
# DIRECTORIES  –  adjust as needed
# =============================================================================

test_dir = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1\Images"
cv_dir   = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_val\seq67\Images"

sw_plot    = False
image_dir  = cv_dir
image_paths = [os.path.join(image_dir, f)
               for f in os.listdir(image_dir)
               if f.endswith(('.png', '.jpg', '.jpeg'))]

# =============================================================================
# OPTIMISE WEIGHTS  (comment this block out to use hand-tuned values directly)
# =============================================================================

# Use a subset of images for fast optimisation; set to image_paths for full run
optimisation_images = image_paths[:1]

opt_class_matrix, opt_threshold_matrix, loss_history = optimise_weights(
    image_paths        = optimisation_images,
    clip_prompts       = clip_prompts,
    mapping_keys       = mapping_keys,
    init_class_matrix  = scale_class_matrix,
    init_threshold_matrix = scale_threshold_matrix,
    num_epochs = 30,
    lr         = 5e-2,
    temperature = 1.0,
)

print("\nOptimised scale_class_matrix:")
for p, d in opt_class_matrix.items():
    print(f"  {p}: {d}")
print("\nOptimised scale_threshold_matrix:")
for p, d in opt_threshold_matrix.items():
    print(f"  {p}: {d}")

# Plot loss curve
plt.figure(figsize=(7, 4))
plt.plot(loss_history, marker='o')
plt.title("GT Cross-Entropy Loss During Weight Optimisation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()

# =============================================================================
# INFERENCE  –  run with optimised (or original) weights
# =============================================================================

miou_lst         = []
weighted_miou_lst = []
per_class_lst    = []
time_lst         = []

for image_path in image_paths:
    start_time = time.time()

    fused_class_map, fused_conf_map, _ = exact_multi_scale_ensemble_matrix(
        image_path,
        clip_prompts,
        mapping_keys,
        opt_class_matrix,                   # <-- use optimised weights
        scale_threshold_matrix = opt_threshold_matrix,
        temperature = 1.0,
        sw_plot     = sw_plot
    )

    results, miou, weighted_miou = save_and_evaluate_single_image(
        image_path  = image_path,
        seg_map     = fused_class_map,
        mapping_keys = mapping_keys
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

    dataframes = [pd.DataFrame(run).T for run in per_class_lst]
    avg_df     = sum(dataframes) / len(dataframes)

    print(f"\n{'='*84}")
    print(f"Image: Average Results")
    print(f"{'='*84}")

    col_w = 20
    print(f"\n{'Category':<{col_w}} {'IoU':>8} {'F0.5':>8} {'F1':>8} {'F2':>8} "
          f"{'Precision':>10} {'Recall':>8}")
    print("-" * 76)

    for name, m in avg_df.iterrows():
        print(f"{name:<{col_w}} {m['IoU']:>8.4f} {m['F0.5']:>8.4f} {m['F1']:>8.4f} "
              f"{m['F2']:>8.4f} {m['Precision']:>10.4f} {m['Recall']:>8.4f}")

    print("-" * 76)
    avg_miou          = avg_df['IoU'].mean()
    avg_weighted_miou = np.mean(weighted_miou_lst)
    print(f"\nmIoU:          {avg_miou:.4f}  ({avg_miou*100:.2f}%)")
    print(f"Weighted mIoU: {avg_weighted_miou:.4f}  ({avg_weighted_miou*100:.2f}%)\n")