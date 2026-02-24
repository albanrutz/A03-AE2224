import torch
import clip
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image
from tqdm import tqdm # For a progress bar
"""
Traditional segmentation + vanilla CLIP approach 
"""
def sliding_window_segmentation(image_path, labels, patch_size=112):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ViT-B/32 is much faster for brute-force patching than RN50
    model, preprocess = clip.load("ViT-B/32", device=device)

    # 1. Load Image and define the strict mathematical grid
    original_image = Image.open(image_path).convert("RGB")
    W, H = original_image.size
    
    # Calculate how many full nxn patches fit into the image
    cols = W // patch_size
    rows = H // patch_size
    
    # Crop the image to exactly fit the grid (removing fractional edges)
    cropped_W, cropped_H = cols * patch_size, rows * patch_size
    original_image = original_image.crop((0, 0, cropped_W, cropped_H))
    
    # Initialize the discrete output map
    segmentation_map = np.zeros((rows, cols), dtype=np.int32)

    # 2. Encode Text Labels once
    text_tokens = clip.tokenize(labels).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    print(f"Extracting {rows * cols} patches of size {patch_size}x{patch_size}...")

    # 3. The Sliding Window Extraction
    patches = []
    coordinates = []
    
    for r in range(rows):
        for c in range(cols):
            # Define the physical bounding box for this patch
            left = c * patch_size
            upper = r * patch_size
            right = left + patch_size
            lower = upper + patch_size
            
            # Extract and preprocess
            patch = original_image.crop((left, upper, right, lower))
            patch_tensor = preprocess(patch)
            
            patches.append(patch_tensor)
            coordinates.append((r, c))

    # Stack all patches into a single batch tensor for hardware acceleration
    # Shape: [Total_Patches, 3, n, n]
    batch_tensor = torch.stack(patches).to(device)

    # 4. Batch Inference (The Math)
    batch_size = 64 # Process in chunks to avoid GPU out-of-memory errors
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(batch_tensor), batch_size), desc="Classifying Patches"):
            chunk = batch_tensor[i : i + batch_size]
            
            # Encode image chunk
            image_features = model.encode_image(chunk)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarities
            similarities = image_features @ text_features.T
            
            # Get the winning label index for each patch in the chunk
            winners = similarities.argmax(dim=-1).cpu().numpy()
            predictions.extend(winners)

    # 5. Reassemble the Grid
    for (r, c), winner_idx in zip(coordinates, predictions):
        segmentation_map[r, c] = winner_idx

    # 6. Visualization
    original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    # Scale the grid back up to the image resolution using Nearest Neighbor
    segmentation_resized = cv2.resize(
        segmentation_map, 
        (cropped_W, cropped_H), 
        interpolation=cv2.INTER_NEAREST
    )

    colors = plt.cm.get_cmap('tab10', len(labels))
    cmap = ListedColormap(colors.colors)

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    ax[0].imshow(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"Original Cropped ({cropped_W}x{cropped_H})")
    ax[0].axis('off')

    # Draw the grid lines to visualize the sliding window physics
    for r in range(rows + 1):
        ax[0].axhline(r * patch_size, color='white', linewidth=0.5, alpha=0.5)
    for c in range(cols + 1):
        ax[0].axvline(c * patch_size, color='white', linewidth=0.5, alpha=0.5)

    ax[1].imshow(segmentation_resized, cmap=cmap, vmin=0, vmax=len(labels)-1, alpha=0.6)
    ax[1].imshow(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB), alpha=0.4)
    ax[1].set_title(f"{patch_size}x{patch_size} Patch Segmentation")
    ax[1].axis('off')

    patches_legend = [mpatches.Patch(color=colors(i), label=labels[i]) for i in range(len(labels))]
    ax[1].legend(handles=patches_legend, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

# --- Execution Example ---
"""
uavid_labels = [
    "this is a segment of an aerial view of a building", 
    "this is a segment of an aerial view of a road", 
    #"this is a segment of an aerial view of a static car", 
    "this is a segment of an aerial view of a tree",
    "this is a segment of an aerial view of low vegetation",
    #"a segment of an aerial view of background clutter"
]
"""
uavid_labels = [
    "aerial view of a building", 
    "aerial view of road with no cars", 
    #"this is a segment of an aerial view of a static car", 
    "aerial view of a tree",
    "aerial view of low vegetation",
    "aerial view of background clutter",
    "aerial view of road with cars"
]

sliding_window_segmentation(r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_test\seq22\Images\000000.png", uavid_labels)
sliding_window_segmentation(r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1\Images\file1-5.png", uavid_labels)
sliding_window_segmentation(r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1\Images\file3-8.png", uavid_labels)
sliding_window_segmentation(r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1\Images\file14-2.png", uavid_labels)