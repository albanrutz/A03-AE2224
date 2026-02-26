import torch
import clip
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image
"""
Another Magic in CLIP hidden layers, Doesn't work very well
"""

def generate_vit_segmentation_map(image_path, labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the Vision Transformer CLIP model
    model, preprocess = clip.load("ViT-B/16", device=device)

    # 1. Prepare Inputs
    original_image = Image.open(image_path).convert("RGB")
    image_input = preprocess(original_image).unsqueeze(0).to(device)
    image_input = image_input.type(model.dtype)

    text_tokens = clip.tokenize(labels).to(device)

    with torch.no_grad():
        # 2. Encode Text Labels (Projected to embedding space)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 3. Manual Forward Pass of the Image Encoder
        # We step through the layers manually to avoid the pooling step that destroys spatial data
        x = model.visual.conv1(image_input) 
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1) 
        
        # Add class embedding and positional embeddings
        x = torch.cat([model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + model.visual.positional_embedding.to(x.dtype)
        x = model.visual.ln_pre(x)
        
        # Pass through the Transformer blocks
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = model.visual.ln_post(x)

        # 4. The Crucial Step: Projecting Spatial Patches
        # Apply the final projection matrix to ALL tokens, not just the [CLS] token
        if model.visual.proj is not None:
            x = x @ model.visual.proj

        # Isolate the 196 spatial patches (drop the [CLS] token at index 0)
        # Shape: [1, 196, 512]
        patch_features = x[:, 1:, :] 
        
        # Normalize the patch vectors to the hypersphere
        patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)

        # 5. Compute the Similarity Matrix
        # Dot product between [196, 512] and [Num_Labels, 512]^T -> [196, Num_Labels]
        similarities = patch_features[0] @ text_features.T
        
        # Apply temperature scaling and Softmax to get probabilities
        logit_scale = model.logit_scale.exp()
        probs = (similarities * logit_scale).softmax(dim=-1)

        # Find the winning label index for each patch
        winner_indices = probs.argmax(dim=-1) # Shape: [196]
        
        # Reshape the 196 indices back into the 2D spatial grid (14x14)
        # Force float32 for safety, though it's discrete integers here
        segmentation_map = winner_indices.reshape(14, 14).cpu().numpy().astype(np.int32)

    # 6. Matplotlib Visualization & Projection
    original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    
    # Upsample the 14x14 grid to the original UAV image dimensions using NEAREST neighbor 
    # (Important: Do not use bilinear interpolation for discrete label indices!)
    segmentation_resized = cv2.resize(
        segmentation_map, 
        (original_cv.shape[1], original_cv.shape[0]), 
        interpolation=cv2.INTER_NEAREST
    )

    # Create a distinct colormap for the labels
    colors = plt.cm.get_cmap('tab10', len(labels))
    cmap = ListedColormap(colors.colors)

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot Original
    ax[0].imshow(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original UAV Image")
    ax[0].axis('off')

    # Plot Segmentation Map
    im = ax[1].imshow(segmentation_resized, cmap=cmap, vmin=0, vmax=len(labels)-1, alpha=0.6)
    ax[1].imshow(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB), alpha=0.4) # Overlay
    ax[1].set_title("Zero-Shot Dense Patch Segmentation")
    ax[1].axis('off')

    # Create a legend
    patches = [mpatches.Patch(color=colors(i), label=labels[i]) for i in range(len(labels))]
    ax[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.show()

# --- Execution Example ---
uavid_labels = [
    "an aerial view of a road", 
    "an aerial view of a building", 
    "an aerial view of vegetation", 
    "an aerial view of a car"
]
# Run the function
generate_vit_segmentation_map(r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_test\seq22\Images\000000.png", uavid_labels)