import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def extract_geometric_features(image_path, sam_checkpoint="sam_vit_b_01ec64.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load the SAM Model (using the ViT-Base architecture)
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # 2. Initialize the Automatic Mask Generator
    # We configure this to filter out mathematically insignificant noise (tiny polygons)
    # which is crucial for high-altitude oblique imagery.
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,          # Density of the point grid
        pred_iou_thresh=0.86,        # Confidence threshold for the boundary
        stability_score_thresh=0.92, # Stability under mathematical perturbation
        crop_n_layers=1,             # Helps with scale variance in aerial shots
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100     # Drop polygons smaller than 10x10 pixels
    )
    
    # 3. Process the Image Matrix
    # OpenCV loads in BGR, SAM expects RGB
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    print("Computing geometric boundaries. This involves a heavy forward pass...")
    # Generate the masks. This returns a list of dictionaries.
    masks = mask_generator.generate(image_rgb)
    print(f"SAM identified {len(masks)} distinct geometric features.")
    
    # Sort masks by area (largest to smallest) to prioritize roads/buildings over cars
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    
    # 4. Extract Isolated Features (Zeroing out the background)
    extracted_features = []
    
    for i, mask_data in enumerate(masks):
        # The mask is a boolean matrix (H x W) where True = object, False = background
        bool_mask = mask_data['segmentation']
        
        # Create an empty black image (matrix of zeros)
        isolated_feature = np.zeros_like(image_rgb)
        
        # Apply the boolean mask: Copy only the pixels belonging to the object
        isolated_feature[bool_mask] = image_rgb[bool_mask]
        
        # To optimize the downstream CLIP step, we crop the matrix to the bounding box
        # rather than feeding the entire 4K black image.
        bbox = mask_data['bbox'] # [x, y, width, height]
        x, y, w, h = [int(v) for v in bbox]
        cropped_feature = isolated_feature[y:y+h, x:x+w]
        
        extracted_features.append({
            'cropped_image': cropped_feature,
            'full_mask': bool_mask,
            'area': mask_data['area']
        })
        
    return image_rgb, extracted_features, masks

def visualize_sam_outputs(image_rgb, extracted_features):
    """
    Visualizes the raw image, the overarching boolean mask map, 
    and a few isolated geometric features.
    """
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    # Show Original
    axs[0, 0].imshow(image_rgb)
    axs[0, 0].set_title("Original Aerial View")
    axs[0, 0].axis('off')
    
    # Combine all boolean masks into one colorful map for visualization
    combined_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4))
    for mask_data in extracted_features:
        m = mask_data['full_mask']
        color_mask = np.concatenate([np.random.random(3), [0.35]]) # Random color + alpha
        combined_mask[m] = color_mask
        
    axs[0, 1].imshow(image_rgb)
    axs[0, 1].imshow(combined_mask)
    axs[0, 1].set_title("SAM Geometric Topography")
    axs[0, 1].axis('off')
    
    # Show the first 4 extracted features (e.g., isolated buildings/roads)
    for i in range(4):
        if i < len(extracted_features):
            row = 1 if i >= 2 else 0
            col = (i % 2) + (1 if row == 1 else 2)
            if row == 0 and col == 2: ax = axs[0, 2]
            elif row == 1 and col == 1: ax = axs[1, 0]
            elif row == 1 and col == 2: ax = axs[1, 1]
            elif row == 1 and col == 3: ax = axs[1, 2] # Adjust index logic for 2x3 grid
            
            # Using a simpler hardcoded assignment for the 4 slots:
            target_axs = [axs[0, 2], axs[1, 0], axs[1, 1], axs[1, 2]]
            target_axs[i].imshow(extracted_features[i]['cropped_image'])
            target_axs[i].set_title(f"Isolated Feature {i+1}\nArea: {extracted_features[i]['area']}px")
            target_axs[i].axis('off')
            
    plt.tight_layout()
    plt.show()

# --- Execution ---
image_rgb, features, raw_masks = extract_geometric_features(r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1\Images\file14-2.png")
visualize_sam_outputs(image_rgb, features)