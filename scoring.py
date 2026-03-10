"""
UAVid Semantic Segmentation Evaluation Script
==============================================
Computes per-class IoU, mIoU, and F0.5 / F1 / F2 scores by comparing
a model output image against a UAVid ground-truth label image.

Both images must use the UAVid colour-mask format (RGB PNG).

Set GT_IMAGE_PATH and PRED_IMAGE_PATH below, then run:
    python evaluate_segmentation.py
"""

import os
import numpy as np
from PIL import Image

# =============================================================================
# INPUT FILE PATHS — set these before running
# =============================================================================

GT_IMAGE_PATH   = "C:\\CLIP_UAVid\\data\\uavid_val\\seq16\\Labels\\000000.png"   # Path to the ground-truth UAVid label image
PRED_IMAGE_PATH = "C:\\CLIP_UAVid\\data\\uavid_val\\seq16\\Labels\\000100.png"   # Path to the model output image

# =============================================================================
# CATEGORY COLOUR MAP
# Fill in the hex colour codes that correspond to each UAVid category.
# Format: "Category Name": (R, G, B)
# The hex code #RRGGBB converts to (0xRR, 0xGG, 0xBB).
# =============================================================================

CATEGORY_COLOURS = {
    "Building":           (128, 0, 0),   # e.g. (128, 0, 0)   — fill in hex: #______
    "Road":               (128, 64, 128),   # e.g. (128, 64, 128) — fill in hex: #______
    "Static Car":         (192, 0, 192),   # e.g. (192, 0, 192)  — fill in hex: #______
    "Tree":               (0, 128, 0),   # e.g. (0, 128, 0)    — fill in hex: #______
    "Low Vegetation":     (128, 128, 0),   # e.g. (128, 128, 0)  — fill in hex: #______
    "Human":              (64, 64, 0),   # e.g. (64, 0, 0)     — fill in hex: #______
    "Moving Car":         (64, 0, 128),   # e.g. (0, 0, 192)    — fill in hex: #______
    "Background Clutter": (0, 0, 0),   # e.g. (0, 0, 0)      — fill in hex: #______
}

# =============================================================================
# OPTIONAL: MERGE STATIC CAR AND MOVING CAR INTO ONE CATEGORY
# Set to True to treat both car types as a single "Car" class.
# When enabled, both colours map to the same index and results are
# reported under "Car" instead of the two separate categories.
# =============================================================================

MERGE_CARS = True

# =============================================================================
# END OF CONFIGURATION — no changes needed below this line
# =============================================================================


def hex_to_rgb(hex_str):
    """Convert a hex string like '#FF00AA' or 'FF00AA' to an (R, G, B) tuple."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def validate_colour_map(colour_map):
    """Check that all category colours have been filled in."""
    missing = [name for name, colour in colour_map.items() if colour is None]
    if missing:
        raise ValueError(
            f"The following categories have no colour assigned:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nPlease fill in CATEGORY_COLOURS at the top of the script."
        )


def build_colour_index(colour_map, merge_cars=False):
    """
    Build a lookup from RGB tuple -> category index.
    Also returns the ordered list of category names.

    If merge_cars=True, both "Static Car" and "Moving Car" map to the
    same index and the merged category is labelled "Car".
    """
    if merge_cars:
        # Build a collapsed category list with "Car" in place of both car types
        merged_categories = []
        seen_car = False
        for name in colour_map:
            if name in ("Static Car", "Moving Car"):
                if not seen_car:
                    merged_categories.append("Car")
                    seen_car = True
            else:
                merged_categories.append(name)

        # Assign indices based on merged list
        car_idx = merged_categories.index("Car")
        colour_to_idx = {}
        name_to_idx = {name: idx for idx, name in enumerate(merged_categories)}

        for name, rgb in colour_map.items():
            if name in ("Static Car", "Moving Car"):
                colour_to_idx[rgb] = car_idx
            else:
                colour_to_idx[rgb] = name_to_idx[name]

        return merged_categories, colour_to_idx
    else:
        categories = list(colour_map.keys())
        colour_to_idx = {rgb: idx for idx, (_, rgb) in enumerate(colour_map.items())}
        return categories, colour_to_idx


def image_to_label_array(image_path, colour_to_idx, num_classes):
    """
    Load an RGB label image and convert every pixel to a class index.
    Pixels whose colour does not match any category are assigned index -1.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w, _ = img.shape
    label = np.full((h, w), fill_value=-1, dtype=np.int32)

    for rgb, idx in colour_to_idx.items():
        # Build a boolean mask where all three channels match
        match = (
            (img[:, :, 0] == rgb[0]) &
            (img[:, :, 1] == rgb[1]) &
            (img[:, :, 2] == rgb[2])
        )
        label[match] = idx

    unmatched = np.sum(label == -1)
    if unmatched > 0:
        total = h * w
        print(f"  Warning: {unmatched}/{total} pixels ({100*unmatched/total:.2f}%) "
              f"in '{os.path.basename(image_path)}' did not match any category colour.")

    return label


def compute_confusion_matrix(gt_label, pred_label, num_classes):
    """
    Compute a (num_classes x num_classes) confusion matrix where
    entry [i, j] is the number of pixels whose true class is i
    and predicted class is j.

    Pixels labelled -1 (unknown colour) are ignored in both images.
    """
    valid_mask = (gt_label >= 0) & (pred_label >= 0)
    gt_flat   = gt_label[valid_mask]
    pred_flat = pred_label[valid_mask]

    # Use numpy's bincount for efficiency
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    combined = gt_flat * num_classes + pred_flat
    counts = np.bincount(combined, minlength=num_classes * num_classes)
    confusion = counts.reshape((num_classes, num_classes))
    return confusion


def per_class_metrics(confusion, categories):
    """
    Derive per-class TP, FP, FN, TN from the confusion matrix, then compute
    IoU, F0.5, F1, and F2 for each category.

    For class i:
        TP = confusion[i, i]
        FP = sum of column i  - TP   (predicted as i but actually something else)
        FN = sum of row i     - TP   (actually i but predicted as something else)
        TN = total pixels - TP - FP - FN
    """
    num_classes = len(categories)
    total_pixels = confusion.sum()

    results = {}

    for i, name in enumerate(categories):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp   # column sum minus diagonal
        fn = confusion[i, :].sum() - tp   # row sum minus diagonal
        tn = total_pixels - tp - fp - fn

        # IoU (Jaccard Index)
        denom_iou = tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else 0.0

        # Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F-scores: F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
        def f_score(beta, p, r):
            beta_sq = beta ** 2
            denom = beta_sq * p + r
            return (1 + beta_sq) * p * r / denom if denom > 0 else 0.0

        f05 = f_score(0.5, precision, recall)
        f1  = f_score(1.0, precision, recall)
        f2  = f_score(2.0, precision, recall)

        results[name] = {
            "TP":        int(tp),
            "FP":        int(fp),
            "FN":        int(fn),
            "TN":        int(tn),
            "Precision": precision,
            "Recall":    recall,
            "IoU":       iou,
            "F0.5":      f05,
            "F1":        f1,
            "F2":        f2,
        }

    return results


def compute_miou(per_class_results):
    """Mean IoU across all classes."""
    iou_values = [v["IoU"] for v in per_class_results.values()]
    return np.mean(iou_values)


def print_results(per_class_results, miou, image_name=None):
    """Pretty-print the evaluation results."""
    header = f"\n{'='*80}"
    if image_name:
        header += f"\nImage: {image_name}"
    header += f"\n{'='*80}"
    print(header)

    col_w = 20
    print(f"\n{'Category':<{col_w}} {'IoU':>8} {'F0.5':>8} {'F1':>8} {'F2':>8} "
          f"{'Precision':>10} {'Recall':>8}")
    print("-" * 76)

    for name, m in per_class_results.items():
        print(f"{name:<{col_w}} {m['IoU']:>8.4f} {m['F0.5']:>8.4f} {m['F1']:>8.4f} "
              f"{m['F2']:>8.4f} {m['Precision']:>10.4f} {m['Recall']:>8.4f}")

    print("-" * 76)
    print(f"\nmIoU: {miou:.4f}  ({miou*100:.2f}%)\n")

    # Confusion matrix counts
    # print(f"\n{'Category':<20} {'TP':>12} {'FP':>12} {'FN':>12} {'TN':>12}")
    # print("-" * 68)
    # for name, m in per_class_results.items():
    #     print(f"{name:<20} {m['TP']:>12,} {m['FP']:>12,} {m['FN']:>12,} {m['TN']:>12,}")


def evaluate_pair(gt_path, pred_path, colour_to_idx, categories):
    """Evaluate a single ground-truth / prediction image pair."""
    num_classes = len(categories)

    gt_label   = image_to_label_array(gt_path,   colour_to_idx, num_classes)
    pred_label = image_to_label_array(pred_path, colour_to_idx, num_classes)

    if gt_label.shape != pred_label.shape:
        raise ValueError(
            f"Image size mismatch: GT is {gt_label.shape}, "
            f"prediction is {pred_label.shape}."
        )

    confusion = compute_confusion_matrix(gt_label, pred_label, num_classes)
    per_class = per_class_metrics(confusion, categories)
    miou      = compute_miou(per_class)

    return per_class, miou, confusion


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    validate_colour_map(CATEGORY_COLOURS)
    categories, colour_to_idx = build_colour_index(CATEGORY_COLOURS, merge_cars=MERGE_CARS)

    if MERGE_CARS:
        print("Note: 'Static Car' and 'Moving Car' are merged into 'Car'.")

    per_class, miou, _ = evaluate_pair(
        GT_IMAGE_PATH, PRED_IMAGE_PATH, colour_to_idx, categories
    )
    print_results(per_class, miou, image_name=os.path.basename(PRED_IMAGE_PATH))