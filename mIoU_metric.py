import numpy as np
import cv2
from PIL import Image

def calculate_uavid_iou_mapped(pred_map, gt_image_path, model_labels):
    """
    Validation engine that maps multiple Ground Truth colors to single model classes.
    """
    # 1. Load Ground Truth
    gt_rgb = np.array(Image.open(gt_image_path).convert("RGB"))
    H, W, _ = gt_rgb.shape
    
    # Resize prediction if needed to match GT exactly
    if pred_map.shape != (H, W):
        pred_map = cv2.resize(pred_map.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    # 2. Define the STRICT UAVid RGB Values (as provided)
    uavid_gt_colors = {
        "building":    [110, 27, 21],
        "road":        [115, 70, 122],
        "tree":        [65, 125, 40],
        "low_veg":     [124, 127, 44],
        "clutter":     [0, 0, 0],
        "static_car":  [167, 46, 181],
        "moving_car":  [60, 20, 120],
        "human":       [64, 66, 23]
    }

    # 3. Define the Mapping: Which GT colors belong to which Model Index?
    # Ensure these indices match your model's output indices exactly!
    # model_labels = ["building", "road", "tree", "low vegetation", "clutter", "cars"]
    mapping = {
        0: ["building"],
        1: ["road"],
        2: ["tree"],
        3: ["low_veg"],
        4: ["clutter"],
        5: ["static_car", "moving_car"] # <--- THE MERGE: Both map to index 5
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

