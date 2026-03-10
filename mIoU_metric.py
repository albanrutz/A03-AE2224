import numpy as np
import cv2
from PIL import Image

def calculate_uavid_iou_mapped(pred_map, gt_image_path, model_labels):
    """
    Validation engine that maps multiple Ground Truth colors to single model classes.
    """
    # 1. Load Ground Truth
    gt_rgb = np.array(Image.open(gt_image_path).convert("RGB"))
    H, W = pred_map.shape[:2]
    
    # Slice GT to match prediction map size
    gt_rgb = gt_rgb[:H, :W]

    # 2. Define the STRICT UAVid RGB Values (as provided)
    uavid_gt_colors = {
        "building":    [128, 0, 0],
        "road":        [128, 64, 128],
        "tree":        [0, 128, 0],
        "low_veg":     [128, 128, 0],
        "clutter":     [0, 0, 0],
        "static_car":  [192, 0, 192],
        "moving_car":  [64, 0, 128],
        "human":       [64, 64, 0]
    }

    # 3. Define the Mapping: Which GT colors belong to which Model Index?
    # IMPORTANT: The keys (indices) in this mapping must correspond exactly to your model's output class indices.
    # If they do not match, IoU calculations will be incorrect because predictions and ground truth will be compared for the wrong classes.
    # Overlapping classes (e.g., "tree" and "low_veg" both mapped to index 2) mean that pixels of either GT color will be treated as belonging to the same model class.
    mapping = {
        0: ["building"],
        1: ["road"],
        2: ["tree", "low_veg"],
        3: ["low_veg"],
        4: ["clutter"],
        5: ["static_car", "moving_car"],
        6: ["human"]
    }

    # Create a 2D Ground Truth map matched to MODEL INDICES
    gt_indices = np.full((H, W), -1, dtype=np.int32) # -1 is 'ignore'
    
    for model_idx, gt_keys in mapping.items():
        for key in gt_keys:
            color = uavid_gt_colors[key]
            mask = np.all(gt_rgb == color, axis=-1)
            gt_indices[mask] = model_idx

    # 4. Calculate IoU per Model Class
    print(f"\n{'Model Class':<20} | {'IoU (%)':<10}")
    print("-" * 35)
    
    iou_list = []
    for i, label in enumerate(model_labels):
        # We only calculate IoU for classes defined in our model
        # Intersection: Both model and GT agree on the (potentially merged) class
        intersection = np.logical_and(pred_map == i, gt_indices == i).sum()
        # Union: Total area covered by either model or GT for that class
        union = np.logical_or(pred_map == i, gt_indices == i).sum()
        
        if union == 0:
            iou = 0.0
        else:
            iou = (intersection / union) * 100
        
        iou_list.append(iou)
        print(f"{label.split()[-1].capitalize():<20} | {iou:>8.2f}%")

    mean_iou = np.mean(iou_list)
    print("-" * 35)
    print(f"{'mIoU (Mean)':<20} | {mean_iou:>8.2f}%")
    
    return iou_list, mean_iou

