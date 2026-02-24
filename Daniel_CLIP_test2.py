import torch
import torch.nn.functional as F
import clip
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------
# 1. The Hook: Capturing the Math
# ---------------------------------------------------------
class SpatialLayerHook:
    """
    Attaches to a specific layer in the CNN to capture the forward pass 
    feature map (Activations) and the backward pass sensitivity (Gradients).
    """
    def __init__(self, module):
        self.activations = None
        self.gradients = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activations = output
        # Register a hook on the output tensor to capture the gradient 
        # flowing backward during the autograd backward pass.
        output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad

    def close(self):
        self.hook.remove()

# ---------------------------------------------------------
# 2. The Core Execution Pipeline
# ---------------------------------------------------------
def generate_zero_shot_heatmap(image_path, text_prompt, target_layer_name='layer4'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the RN50 CLIP model
    model, preprocess = clip.load("RN50", device=device)
    
    # Prepare Image and Text inputs
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    text_tensor = clip.tokenize([text_prompt]).to(device)

    # Attach the hook to the last spatial feature map
    # In ResNet50, 'layer4' is the final convolutional block before pooling
    target_layer = getattr(model.visual, target_layer_name)
    hook = SpatialLayerHook(target_layer)

    # --- FORWARD PASS ---
    image_features = model.encode_image(image_tensor)
    text_features = model.encode_text(text_tensor)
    
    # Normalize vectors to the hypersphere
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate the similarity score (dot product)
    # This scalar is the target function we want to maximize
    similarity_score = (image_features * text_features).sum()

    # --- BACKWARD PASS ---
    model.zero_grad()
    similarity_score.backward()

    # --- HEATMAP SYNTHESIS ---
    gradients = hook.gradients     # Dimensions: [1, Channels, Height, Width]
    activations = hook.activations # Dimensions: [1, Channels, Height, Width]

    # Global Average Pooling of gradients to find channel importance weights (alpha)
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

    # Linear combination of activations weighted by the gradient importance
    heatmap = torch.sum(weights * activations, dim=1, keepdim=True)
    
    # Apply ReLU (we only care about positive influences on the target class)
    heatmap = F.relu(heatmap)
    
    # Normalize the heatmap to a [0, 1] range for visual mapping
    heatmap = heatmap.squeeze().cpu().detach().numpy().astype(np.float32)
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    hook.close()

    # ---------------------------------------------------------
    # 3. Projection and Visualization
    # ---------------------------------------------------------
    original_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Upsample the low-res heatmap to the original UAV image size
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Convert heatmap to a color map (Red = High alignment, Blue = Low alignment)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Overlay the heatmap onto the original image
    superimposed_img = cv2.addWeighted(original_img, 0.5, heatmap_colored, 0.5, 0)

    # Plot the results side-by-side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Oblique UAV View")
    plt.axis('off')
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 2, 2)
    plt.title(f"Grad-CAM Activation for: '{text_prompt}'")
    plt.axis('off')
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()
    #plt.savefig(f"heatmap_{text_prompt.replace(' ', '_')}.png")

# --- Execution Example ---
# To test this on your UAVid dataset, simply run:
# generate_zero_shot_heatmap("path/to/uavid_image.png", "an aerial view of a road")

#generate_zero_shot_heatmap(r"C:\Users\danie\Desktop\Delft archive\AE2224\000000.png", "an aerial view of a building")
#generate_zero_shot_heatmap(r"C:\Users\danie\Desktop\Delft archive\AE2224\000000.png", "an aerial view of a tree")
#generate_zero_shot_heatmap(r"C:\Users\danie\Desktop\Delft archive\AE2224\000000.png", "a section of a drone photo of a building")
#generate_zero_shot_heatmap(r"C:\Users\danie\Desktop\Delft archive\AE2224\000000.png", "a section of a drone photo of a tree")
generate_zero_shot_heatmap(r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_test\seq21\Images\000000.png", "an aerial view of a roof of a building")
generate_zero_shot_heatmap(r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_test\seq21\Images\000000.png", "an aerial view of a wall of a building")
generate_zero_shot_heatmap(r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_test\seq21\Images\000000.png", "an aerial view of a road")
generate_zero_shot_heatmap(r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_test\seq21\Images\000000.png", "an aerial view of a tree")
generate_zero_shot_heatmap(r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_test\seq21\Images\000000.png", "an aerial view of a moving car")