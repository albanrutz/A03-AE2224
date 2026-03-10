import os
import numpy as np
from PIL import Image

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_colour_map(colour_map):
    missing = [name for name, colour in colour_map.items() if colour is None]
    if missing:
        raise ValueError(
            f"The following categories have no colour assigned:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

def build_colour_index(colour_map, merge_cars=False):
    if merge_cars:
        merged_categories = []
        seen_car = False
        for name in colour_map:
            if name in ("Static Car", "Moving Car"):
                if not seen_car:
                    merged_categories.append("Car")
                    seen_car = True
            else:
                merged_categories.append(name)

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
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w, _ = img.shape
    label = np.full((h, w), fill_value=-1, dtype=np.int32)

    for rgb, idx in colour_to_idx.items():
        match = (
            (img[:, :, 0] == rgb[0]) &
            (img[:, :, 1] == rgb[1]) &
            (img[:, :, 2] == rgb[2])
        )
        label[match] = idx

    return label

def compute_confusion_matrix(gt_label, pred_label, num_classes):
    valid_mask = (gt_label >= 0) & (pred_label >= 0)
    gt_flat   = gt_label[valid_mask]
    pred_flat = pred_label[valid_mask]

    combined = gt_flat * num_classes + pred_flat
    counts = np.bincount(combined, minlength=num_classes * num_classes)
    confusion = counts.reshape((num_classes, num_classes))
    return confusion

def per_class_metrics(confusion, categories):
    num_classes = len(categories)
    total_pixels = confusion.sum()
    results = {}

    for i, name in enumerate(categories):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp  
        fn = confusion[i, :].sum() - tp  
        tn = total_pixels - tp - fp - fn

        denom_iou = tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        def f_score(beta, p, r):
            beta_sq = beta ** 2
            denom = beta_sq * p + r
            return (1 + beta_sq) * p * r / denom if denom > 0 else 0.0

        results[name] = {
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "Precision": precision, "Recall": recall, "IoU": iou,
            "F0.5": f_score(0.5, precision, recall),
            "F1": f_score(1.0, precision, recall),
            "F2": f_score(2.0, precision, recall),
        }

    return results

def print_results(per_class_results, miou, image_name=None):
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

# =============================================================================
# MAIN CALLABLE FUNCTION
# =============================================================================

def run_uavid_evaluation(gt_path, pred_path, category_colours=None, merge_cars=True, verbose=True):
    """
    Evaluates a single ground-truth / prediction image pair.
    
    Args:
        gt_path (str): Path to the ground truth image.
        pred_path (str): Path to the predicted mask image.
        category_colours (dict, optional): Dictionary mapping class names to RGB tuples.
        merge_cars (bool): Whether to combine 'Static Car' and 'Moving Car'.
        verbose (bool): If True, prints the formatted results to the console.
        
    Returns:
        tuple: (per_class_results_dict, mean_iou_float, confusion_matrix_ndarray)
    """
    # Default fallback if no custom dictionary is provided
    if category_colours is None:
        category_colours = {
            "Building":           (128, 0, 0),
            "Road":               (128, 64, 128),
            "Static Car":         (192, 0, 192),
            "Tree":               (0, 128, 0),
            "Low Vegetation":     (128, 128, 0),
            "Human":              (64, 64, 0),
            "Moving Car":         (64, 0, 128),
            "Background Clutter": (0, 0, 0),
        }

    validate_colour_map(category_colours)
    categories, colour_to_idx = build_colour_index(category_colours, merge_cars=merge_cars)

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
    
    iou_values = [v["IoU"] for v in per_class.values()]
    miou = np.mean(iou_values)

    if verbose:
        if merge_cars:
            print("Note: 'Static Car' and 'Moving Car' are merged into 'Car'.")
        print_results(per_class, miou, image_name=os.path.basename(pred_path))

    return per_class, miou, confusion