"""
optimize_weights_blackmagiclip.py
────────────────────────────────────────────────────────────────────────────────
Adam-based gradient optimizer for BlackMagiCLIP's scale_class_matrix and
scale_threshold_matrix.

DIFFERENCES FROM PatchCLIP OPTIMIZER
──────────────────────────────────────
• Model: RN50 (CNN) instead of ViT-B/32 — required for SpatialLayerHook.
• Signal: Grad-CAM dense heatmaps (H, W, C) are cached per (image, scale),
  not flat (N_patches, C) logit vectors.
• Cache cost: ~H×W×C×4 bytes per (image, scale).  For UAVid at 4 scales this
  is ≈ 800 MB per image — heavy but a one-time cost.  Reduce NUM_CACHE_IMAGES
  or IMAGE_SUBSAMPLE if RAM is tight.
• Fusion module: operates on pre-computed (H, W, C) tensors rather than
  patch logits, so forward passes during training are extremely cheap.
• Grad-CAM is frozen in the cache — we are optimising the *weighting and
  gating* of those spatial maps, not re-running Grad-CAM each epoch.

KEY IDEAS (shared with PatchCLIP optimizer)
────────────────────────────────────────────
1. CLIP + Grad-CAM run ONCE.  Results are cached in RAM.
2. Pipeline is end-to-end differentiable:
     • Weights     → log-space  (exp keeps them ≥ 0)
     • Thresholds  → logit-space (sigmoid keeps them in (0,1))
     • Hard gate   → differentiable sigmoid gate, steepness annealed upward
3. FROZEN ZEROS: any weight/threshold that is 0 in the original matrices
   stays exactly 0 throughout optimisation.
4. After optimisation the learned values are printed copy-paste-ready.

USAGE
─────
    python optimize_weights_blackmagiclip.py

    Adjust IMAGE_DIRS and hyper-parameters at the bottom of this file.
────────────────────────────────────────────────────────────────────────────────
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
from tqdm import tqdm

from scoring_general import (
    CATEGORY_COLOURS,
    validate_colour_map,
    build_colour_index,
    image_to_label_array,
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 – GRAD-CAM HOOK  (identical to BlackMagiCLIP)
# ══════════════════════════════════════════════════════════════════════════════

class SpatialLayerHook:
    def __init__(self, module):
        self.activations = None
        self.gradients   = None
        self.hook        = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activations = output
        self.hook_grad   = output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad

    def close(self):
        self.hook.remove()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – FEATURE CACHING  (Grad-CAM dense maps, one-time)
# ══════════════════════════════════════════════════════════════════════════════

def _gt_to_clip_index_map(gt_categories: list, mapping_keys: list) -> list:
    """
    Build gt_to_clip[gt_idx] = clip_idx with the same alias table as
    PatchCLIP optimizer.
    """
    _ALIASES = {
        "background_clutter": "clutter",
        "low_vegetation":     "low_veg",
        "static_car":         "car",
        "moving_car":         "car",
    }

    def _norm(s):
        s = s.lower().replace(" ", "_")
        return _ALIASES.get(s, s)

    clip_lookup = {k: i for i, k in enumerate(mapping_keys)}
    result = []
    for name in gt_categories:
        n = _norm(name)
        if n not in clip_lookup:
            raise ValueError(
                f"GT category '{name}' (normalised: '{n}') cannot be mapped "
                f"to any of the CLIP keys: {mapping_keys}"
            )
        result.append(clip_lookup[n])
    return result


def _compute_gradcam_for_scale(
    model,
    hook: SpatialLayerHook,
    preprocess,
    original_image: Image.Image,
    text_features: torch.Tensor,
    mapping_keys: list,
    scale_weights_dict: dict,
    scale_thresholds_dict: dict,
    p_size: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    Run Grad-CAM for ONE (image, scale) pair.
    Returns a dense float32 array of shape (H_orig, W_orig, C).

    This replicates BlackMagiCLIP's inner loop exactly, but without the
    visualisation block and with an explicit threshold array for the hard gate.
    """
    W_orig, H_orig = original_image.size
    num_classes    = len(mapping_keys)

    current_threshold_array = torch.tensor(
        [scale_thresholds_dict.get(k, 0.0) for k in mapping_keys],
        device=device, dtype=torch.float32,
    )

    pad_w    = (p_size - (W_orig % p_size)) % p_size
    pad_h    = (p_size - (H_orig % p_size)) % p_size
    padded_w = W_orig + pad_w
    padded_h = H_orig + pad_h

    padded_img = Image.new("RGB", (padded_w, padded_h), color=(0, 0, 0))
    padded_img.paste(original_image, (0, 0))

    cols = padded_w // p_size
    rows = padded_h // p_size

    patches, boxes = [], []
    for r in range(rows):
        for c in range(cols):
            left, upper = c * p_size, r * p_size
            right, lower = left + p_size, upper + p_size
            patches.append(preprocess(padded_img.crop((left, upper, right, lower))))
            boxes.append((upper, lower, left, right))

    batch_tensor = torch.stack(patches).to(device)

    # Active classes (weight > 0) — only run backprop for these
    active_classes = [
        i for i, k in enumerate(mapping_keys)
        if scale_weights_dict.get(k, 1.0) > 0.0
    ]

    padded_scale_tensor = np.zeros((padded_h, padded_w, num_classes), dtype=np.float32)

    for i in tqdm(range(0, len(batch_tensor), batch_size), desc=f"  Grad-CAM {p_size}px", leave=False):
        chunk = batch_tensor[i : i + batch_size].type(model.dtype)
        chunk.requires_grad = True

        image_features = model.encode_image(chunk).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits  = model.logit_scale.exp().float() * (image_features @ text_features.T)
        probs   = torch.softmax(logits, dim=-1)

        # Hard activation gate (same as BlackMagiCLIP)
        probs = torch.where(probs < current_threshold_array, torch.zeros_like(probs), probs)

        batch_dense_cams = torch.zeros(
            (chunk.shape[0], num_classes, p_size, p_size), device=device
        )

        for c_idx in active_classes:
            if probs[:, c_idx].sum() == 0:
                continue

            model.zero_grad()
            score = (image_features @ text_features[c_idx])
            score.sum().backward(retain_graph=True)

            if hook.gradients is not None:
                g   = hook.gradients.clone().float()
                a   = hook.activations.clone().float()
                w   = torch.mean(g, dim=[2, 3], keepdim=True)
                cam = F.relu(torch.sum(w * a, dim=1, keepdim=True))   # (B,1,7,7)

                cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
                cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
                cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)
                cam_up   = F.interpolate(cam_norm, size=(p_size, p_size), mode='nearest')

                class_prob = probs[:, c_idx].view(-1, 1, 1, 1)
                batch_dense_cams[:, c_idx, :, :] = (cam_up * class_prob).squeeze(1)

        chunk_cams = batch_dense_cams.permute(0, 2, 3, 1).detach().cpu().numpy()
        for j, (u, l, left, right) in enumerate(boxes[i : i + batch_size]):
            padded_scale_tensor[u:l, left:right, :] = chunk_cams[j]

        del chunk, image_features, logits, probs, batch_dense_cams
        torch.cuda.empty_cache()

    return padded_scale_tensor[:H_orig, :W_orig, :]   # (H_orig, W_orig, C)


def precompute_gradcam_cache(
    image_paths:            list,
    clip_prompts:           list,
    mapping_keys:           list,
    scale_class_matrix:     dict,
    scale_threshold_matrix: dict,
    colour_to_idx:          dict,
    num_classes_gt:         int,
    device:                 str,
    batch_size:             int = 32,
) -> list:
    """
    Run Grad-CAM for every (image, scale) pair and cache the dense maps.

    Each cache entry:
        {
          "gradcam_per_scale": { p_size: np.ndarray(H, W, C_clip) },
          "gt_map":            np.ndarray(H, W)   — integer class labels,
          "image_path":        str,
        }

    RAM cost ≈ H × W × C × 4 bytes × #scales per image.
    For UAVid (3840×2160), 7 classes, 2 scales → ~900 MB per image.
    Reduce batch to 1 image or subsample if needed.
    """
    model, preprocess = clip.load("RN50", device=device)
    model.eval()
    hook = SpatialLayerHook(model.visual.layer4)

    text_tokens = clip.tokenize(clip_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    patch_scales = sorted(scale_class_matrix.keys(), reverse=True)
    cache        = []

    for image_path in tqdm(image_paths, desc="Caching Grad-CAM maps"):
        gt_path = image_path.replace("Images", "Labels")
        if not os.path.exists(gt_path):
            print(f"  [skip] no GT for {os.path.basename(image_path)}")
            continue

        original_image = Image.open(image_path).convert("RGB")
        gt_label       = image_to_label_array(gt_path, colour_to_idx, num_classes_gt)
        # gt_label: (H, W) int array, -1 for unlabelled pixels

        gradcam_per_scale = {}
        for p_size in patch_scales:
            print(f"\n  [{os.path.basename(image_path)}] Scale {p_size}px")
            gradcam_per_scale[p_size] = _compute_gradcam_for_scale(
                model, hook, preprocess,
                original_image, text_features,
                mapping_keys,
                scale_class_matrix[p_size],
                scale_threshold_matrix.get(p_size, {}),
                p_size, batch_size, device,
            )

        cache.append({
            "gradcam_per_scale": gradcam_per_scale,
            "gt_map":            gt_label,
            "image_path":        image_path,
        })

    hook.close()
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – DIFFERENTIABLE FUSION MODULE
# ══════════════════════════════════════════════════════════════════════════════

class FusionWeights(nn.Module):
    """
    Learnable wrapper around scale_class_matrix and scale_threshold_matrix
    for the BlackMagiCLIP spatial-tensor fusion.

    Parameterisation (identical to PatchCLIP optimizer):
    • Weights    → log-space  (exp ≥ 0, frozen zeros stay 0)
    • Thresholds → logit-space (sigmoid ∈ (0,1), frozen zeros stay 0)

    Forward: takes pre-computed Grad-CAM tensors {p_size: (H,W,C)} and
    produces a fused (H,W,C) probability tensor — differentiably.
    """

    def __init__(
        self,
        scale_class_matrix:     dict,
        scale_threshold_matrix: dict,
        mapping_keys:           list,
    ):
        super().__init__()
        self.mapping_keys = mapping_keys
        self.patch_scales = sorted(scale_class_matrix.keys(), reverse=True)

        for p in self.patch_scales:
            w_init = [scale_class_matrix[p].get(k, 0.0)              for k in mapping_keys]
            t_init = [scale_threshold_matrix.get(p, {}).get(k, 0.0)  for k in mapping_keys]

            w_freeze = torch.tensor([1.0 if w == 0.0 else 0.0 for w in w_init])
            t_freeze = torch.tensor([1.0 if t == 0.0 else 0.0 for t in t_init])
            self.register_buffer(f"w_freeze_{p}", w_freeze)
            self.register_buffer(f"t_freeze_{p}", t_freeze)

            log_w = torch.tensor(
                [math.log(max(w, 1e-6)) for w in w_init], dtype=torch.float32
            )
            self.register_parameter(f"log_w_{p}", nn.Parameter(log_w))

            def _logit(t):
                if t <= 0.0:
                    return -10.0
                t = max(min(t, 1 - 1e-6), 1e-6)
                return math.log(t / (1.0 - t))

            raw_t = torch.tensor([_logit(t) for t in t_init], dtype=torch.float32)
            self.register_parameter(f"raw_t_{p}", nn.Parameter(raw_t))

    # ─── Accessors ────────────────────────────────────────────────────────────

    def weights(self, p_size: int) -> torch.Tensor:
        mask  = getattr(self, f"w_freeze_{p_size}")
        log_w = getattr(self, f"log_w_{p_size}")
        return torch.exp(log_w) * (1.0 - mask)       # (C,)

    def thresholds(self, p_size: int) -> torch.Tensor:
        mask  = getattr(self, f"t_freeze_{p_size}")
        raw_t = getattr(self, f"raw_t_{p_size}")
        return torch.sigmoid(raw_t) * (1.0 - mask)   # (C,)

    # ─── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        gradcam_per_scale: dict,
        gate_steepness:    float = 50.0,
    ) -> torch.Tensor:
        """
        gradcam_per_scale: { p_size: np.ndarray(H, W, C) }
            Pre-computed Grad-CAM maps from the cache.

        Returns: fused Tensor(H, W, C) — raw (un-normalised) scores on device.

        The differentiable gate replaces the hard-zero activation from
        BlackMagiCLIP:
            gate = sigmoid( steepness × (cam_val − threshold) )
        allowing gradients to flow back into the threshold parameters.
        """
        dev = next(self.parameters()).device

        # Grab spatial dims from first available scale
        sample = next(iter(gradcam_per_scale.values()))
        H, W, C = sample.shape
        fused   = torch.zeros(H, W, C, device=dev)

        for p in self.patch_scales:
            # (H, W, C) → torch tensor, move to device
            cam = torch.from_numpy(gradcam_per_scale[p]).to(dev)   # (H, W, C)

            w   = self.weights(p)       # (C,)
            thr = self.thresholds(p)    # (C,)

            # Differentiable soft gate (replaces BlackMagiCLIP's hard gate)
            # cam values are already prob-modulated Grad-CAM ∈ [0, 1]
            gate  = torch.sigmoid(gate_steepness * (cam - thr.view(1, 1, C)))
            gated = cam * gate                                      # (H, W, C)

            fused = fused + gated * w.view(1, 1, C)

        return fused   # (H, W, C)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – DIFFERENTIABLE SOFT mIoU LOSS
# ══════════════════════════════════════════════════════════════════════════════

def soft_miou_loss(
    fused:      torch.Tensor,   # (H, W, C_clip)
    gt_map:     np.ndarray,     # (H, W)  integer GT labels, -1 = ignore
    gt_to_clip: list,           # length C_gt; maps gt_idx → clip_idx
    num_classes_gt: int,
) -> torch.Tensor:
    """
    Pixel-level soft mIoU.

    We flatten spatial dims → (N_valid, C), build a soft GT one-hot from
    ground truth labels, then compute per-class soft IoU exactly as in the
    PatchCLIP optimizer.

    Ignores pixels where gt_map == -1 (unlabelled / void).
    """
    dev = fused.device
    H, W, C = fused.shape

    # Mask out unlabelled pixels
    valid_mask = (gt_map >= 0).flatten()                    # (H*W,)
    valid_idx  = torch.from_numpy(np.where(valid_mask)[0]).to(dev)

    p_flat = fused.view(-1, C)[valid_idx]                   # (N_valid, C)
    # L1-normalise per pixel so values sum to 1
    p_norm = p_flat / (p_flat.sum(dim=-1, keepdim=True) + 1e-8)

    gt_flat   = gt_map.flatten()[valid_mask]                # (N_valid,) numpy
    gt_tensor = torch.from_numpy(gt_flat.astype(np.int64)).to(dev)

    # Build soft GT in CLIP class space (many-to-one allowed via gt_to_clip)
    g = torch.zeros(len(valid_idx), C, device=dev)
    for gt_idx, clip_idx in enumerate(gt_to_clip):
        mask_class = (gt_tensor == gt_idx).float()
        g[:, clip_idx] += mask_class

    # Clamp to [0,1] in case two GT classes map to the same CLIP class
    g = g.clamp(0.0, 1.0)

    iou_per_class = []
    for c in range(C):
        p_c = p_norm[:, c]
        g_c = g[:, c]
        tp  = (p_c * g_c).sum()
        fp  = (p_c * (1.0 - g_c)).sum()
        fn  = ((1.0 - p_c) * g_c).sum()
        iou_per_class.append(tp / (tp + fp + fn + 1e-8))

    return torch.stack(iou_per_class).mean()   # scalar, maximise this


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – MAIN OPTIMISER LOOP
# ══════════════════════════════════════════════════════════════════════════════

def optimize_weights(
    image_paths:            list,
    clip_prompts:           list,
    mapping_keys:           list,
    scale_class_matrix:     dict,
    scale_threshold_matrix: dict,
    *,
    num_epochs:      int   = 300,
    lr:              float = 3e-2,
    steepness_start: float = 10.0,
    steepness_end:   float = 80.0,
    print_every:     int   = 10,
    batch_size:      int   = 32,    # Grad-CAM batch — keep ≤32 for RTX 4070
):
    """
    Entry-point.  Returns (optimized_scale_class_matrix, optimized_scale_threshold_matrix)
    in the exact same nested-dict format used by BlackMagiCLIP.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {device}")

    # ── GT setup ─────────────────────────────────────────────────────────────
    validate_colour_map(CATEGORY_COLOURS)
    gt_categories, colour_to_idx = build_colour_index(
        CATEGORY_COLOURS, merge_cars=True, merge_vegetation=False
    )
    num_classes_gt = len(gt_categories)
    print(f"[*] GT categories ({num_classes_gt}): {gt_categories}")

    gt_to_clip = _gt_to_clip_index_map(gt_categories, mapping_keys)
    print(f"[*] GT→CLIP map: {list(zip(gt_categories, [mapping_keys[i] for i in gt_to_clip]))}")

    # ── Feature cache (Grad-CAM forward, one-time) ────────────────────────────
    cache = precompute_gradcam_cache(
        image_paths, clip_prompts, mapping_keys,
        scale_class_matrix, scale_threshold_matrix,
        colour_to_idx, num_classes_gt, device, batch_size=batch_size,
    )
    if not cache:
        raise RuntimeError("No valid (image, GT) pairs found. Check IMAGE_DIRS.")
    print(f"\n[*] Cached Grad-CAM maps for {len(cache)} image(s).\n")

    # ── Model + optimiser ─────────────────────────────────────────────────────
    model     = FusionWeights(scale_class_matrix, scale_threshold_matrix, mapping_keys).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    best_miou  = -1.0
    best_state = None

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        t_frac    = (epoch - 1) / max(num_epochs - 1, 1)
        steepness = steepness_start + t_frac * (steepness_end - steepness_start)

        optimizer.zero_grad()

        total_miou = 0.0
        for item in cache:
            fused    = model(item["gradcam_per_scale"], gate_steepness=steepness)
            miou_val = soft_miou_loss(fused, item["gt_map"], gt_to_clip, num_classes_gt)
            (-miou_val).backward()
            total_miou += miou_val.item()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        avg_miou = total_miou / len(cache)

        if avg_miou > best_miou:
            best_miou  = avg_miou
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % print_every == 0:
            print(
                f"  Epoch {epoch:4d}/{num_epochs}  |  "
                f"Soft mIoU: {avg_miou:.4f}  |  "
                f"Best: {best_miou:.4f}  |  "
                f"Steepness: {steepness:.1f}  |  "
                f"LR: {scheduler.get_last_lr()[0]:.5f}"
            )

    # ── Extract results ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    patch_scales = sorted(scale_class_matrix.keys(), reverse=True)
    return _extract_matrices(model, mapping_keys, patch_scales,
                              scale_class_matrix, scale_threshold_matrix)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – RESULT EXTRACTION + PRETTY-PRINT
# ══════════════════════════════════════════════════════════════════════════════

def _extract_matrices(model, mapping_keys, patch_scales, orig_w, orig_t):
    """Return optimised matrices in the original nested-dict format."""
    new_w, new_t = {}, {}
    for p in patch_scales:
        w_vals = model.weights(p).detach().cpu().tolist()
        t_vals = model.thresholds(p).detach().cpu().tolist()

        orig_w_row = orig_w[p]
        orig_t_row = orig_t.get(p, {})

        new_w[p] = {
            k: (round(w, 6) if orig_w_row.get(k, 0.0) != 0.0 else 0.0)
            for k, w in zip(mapping_keys, w_vals)
        }
        new_t[p] = {
            k: (round(t, 6) if orig_t_row.get(k, 0.0) != 0.0 else 0.0)
            for k, t in zip(mapping_keys, t_vals)
        }
    return new_w, new_t


def print_optimised_matrices(new_w, new_t):
    print("\n" + "═" * 72)
    print("  OPTIMISED  scale_class_matrix  (copy-paste into BlackMagiCLIP)")
    print("═" * 72)
    print("scale_class_matrix = {")
    for p, d in sorted(new_w.items(), reverse=True):
        print(f"    {p}: {{")
        for k, v in d.items():
            print(f'        "{k}": {v:.4f},')
        print("    },")
    print("}")

    print("\n" + "═" * 72)
    print("  OPTIMISED  scale_threshold_matrix")
    print("═" * 72)
    print("scale_threshold_matrix = {")
    for p, d in sorted(new_t.items(), reverse=True):
        active = {k: v for k, v in d.items() if v > 1e-5}
        if active:
            print(f"    {p}: {{")
            for k, v in d.items():
                print(f'        "{k}": {v:.4f},')
            print("    },")
    print("}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Copy your setup from BlackMagiCLIP ───────────────────────────────────
    
    clip_prompts = [
        "drone view of a building",
        "drone view of a road",
        "drone view of a tree",
        "drone view of low vegetation",
        "drone view of background clutter",
        "drone view of a car",
        "drone view of a human",
    ]

    mapping_keys = ["building", "road", "tree", "low_veg", "clutter", "car", "human"]

    scale_class_matrix = {
        448: {
            "building": 1.1,
            "road":     1.5,
            "tree":     1.2,
            "low_veg":  1.0,
            "clutter":  1.2,
            "car":      0.0,   # frozen zero — will not be optimised
            "human":    0.0,   # frozen zero — will not be optimised
        },
        224: {
            "building": 1.0,
            "road":     1.3,
            "tree":     1.1,
            "low_veg":  1.0,
            "clutter":  1.2,
            "car":      1.0,
            "human":    1.2,
        },
    }

    scale_threshold_matrix = {
        448: {
            "building": 0.0,
            "road":     0.0,
            "tree":     0.0,
            "low_veg":  0.0,
            "clutter":  0.0,
            "car":      0.0,
            "human":    0.0,
        },
        224: {
            "building": 0.5,
            "road":     0.0,
            "tree":     0.5,
            "low_veg":  0.4,
            "clutter":  0.1,
            "car":      0.85,
            "human":    0.75,
        },
    }

    # ── Point at your validation images ──────────────────────────────────────
    IMAGE_DIRS = [
        r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_val\seq67\Images",
        # r"C:\...\seq16\Images",
    ]

    # NOTE: Grad-CAM caching is slow and memory-heavy.
    # Start with 1–3 images; add more once you've confirmed the pipeline runs.
    # Set to None to use all images found.
    NUM_CACHE_IMAGES = 13

    image_paths = []
    for d in IMAGE_DIRS:
        image_paths += [
            os.path.join(d, f) for f in os.listdir(d)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    if NUM_CACHE_IMAGES is not None:
        image_paths = image_paths[:NUM_CACHE_IMAGES]
    print(f"[*] Using {len(image_paths)} image(s) for optimisation.")

    # ── Run optimisation ──────────────────────────────────────────────────────
    opt_w, opt_t = optimize_weights(
        image_paths            = image_paths,
        clip_prompts           = clip_prompts,
        mapping_keys           = mapping_keys,
        scale_class_matrix     = scale_class_matrix,
        scale_threshold_matrix = scale_threshold_matrix,
        num_epochs             = 300,
        lr                     = 3e-2,
        steepness_start        = 10.0,
        steepness_end          = 80.0,
        print_every            = 10,
        batch_size             = 32,    # lower to 8–16 if you hit OOM
    )

    print_optimised_matrices(opt_w, opt_t)