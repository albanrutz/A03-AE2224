import torch
import torch.nn.functional as F
import clip
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from tqdm import tqdm 

class SpatialLayerHook:
    def __init__(self, module):
        self.activations = None
        self.gradients = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activations = output
        # Capture gradients during backward pass
        self.hook_grad = output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad

    def close(self):
        self.hook.remove()

def sliding_window_gradcam(image_path, text_prompt="an aerial view of a building", patch_size=224):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load RN50 (ResNet version of CLIP)
    model, preprocess = clip.load("RN50", device=device)
    model.eval() # Ensure we are in eval mode for Grad-CAM

    # 1. Load and Slice Image
    original_image = Image.open(image_path).convert("RGB")
    W, H = original_image.size
    cols, rows = W // patch_size, H // patch_size
    
    # 2. Attach Hook ONCE outside the loop
    target_layer = model.visual.layer4
    hook = SpatialLayerHook(target_layer)

    # 3. Prepare Text Vector
    text_tensor = clip.tokenize([text_prompt]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tensor)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # 4. Iterate through the grid
    for r in range(rows):
        for c in range(cols):
            left, upper = c * patch_size, r * patch_size
            right, lower = left + patch_size, upper + patch_size
            
            # Extract and preprocess
            patch = original_image.crop((left, upper, right, lower))
            # Critical: Cast to model.dtype (float16) for your 4070
            image_tensor = preprocess(patch).unsqueeze(0).to(device).type(model.dtype)

            # --- FORWARD PASS ---
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Calculate similarity
            similarity_score = (image_features * text_features).sum()

            # --- BACKWARD PASS ---
            model.zero_grad()
            similarity_score.backward(retain_graph=True)

            # --- HEATMAP GENERATION ---
            gradients = hook.gradients 
            activations = hook.activations 

            # Calculate channel weights
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
            heatmap = torch.sum(weights * activations, dim=1, keepdim=True)
            heatmap = F.relu(heatmap)

            # Normalize and Convert to NumPy
            heatmap_np = heatmap.squeeze().cpu().detach().numpy().astype(np.float32)
            heatmap_np = (heatmap_np - np.min(heatmap_np)) / (np.max(heatmap_np) + 1e-8)

            # --- VISUALIZATION ---
            original_patch_cv = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2BGR)
            heatmap_resized = cv2.resize(heatmap_np, (patch_size, patch_size))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            
            superimposed = cv2.addWeighted(original_patch_cv, 0.5, heatmap_colored, 0.5, 0)

            # Show every patch (Warning: this will pop up MANY windows if the image is big)
            # Consider adding a "plt.close()" or saving to disk instead
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(np.array(patch))
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
            plt.title(f"Score: {similarity_score.item():.3f}")
            plt.axis('off')
            plt.show()

    hook.close()

# Run it!
sliding_window_gradcam(r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1\Images\file14-2.png")