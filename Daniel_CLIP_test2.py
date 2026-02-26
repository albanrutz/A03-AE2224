import torch
import torch.nn.functional as F
import clip
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

class SpatialLayerHook:
    def __init__(self, module):
        self.activations = None
        self.gradients = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activations = output
        self.hook_grad = output.register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad

    def close(self):
        self.hook.remove()

def analyze_all_patches_gradcam(image_path, labels, patch_size=224):
    # 1. Hardware Initialization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)
    model.eval()

    # 2. Image Slicing Setup
    original_image = Image.open(image_path).convert("RGB")
    W, H = original_image.size
    cols, rows = W // patch_size, H // patch_size
    
    # 3. Setup Hook and Text Encodings
    target_layer = model.visual.layer4
    hook = SpatialLayerHook(target_layer)

    text_tokens = clip.tokenize(labels).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    print(f"Starting analysis for {rows * cols} patches...")

    # 4. Main Loop across the Grid
    for r in range(rows):
        for c in range(cols):
            left, upper = c * patch_size, r * patch_size
            right, lower = left + patch_size, upper + patch_size
            patch = original_image.crop((left, upper, right, lower))
            
            # Prepare image tensor (float16 for 4070)
            image_tensor = preprocess(patch).unsqueeze(0).to(device).type(model.dtype)

            # --- FORWARD PASS ---
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Get probabilities
            logit_scale = model.logit_scale.exp()
            probs = (logit_scale * (image_features @ text_features.T)).softmax(dim=-1).detach().cpu().numpy()[0]

            # --- MULTI-LABEL BACKWARD PASSES ---
            patch_heatmaps = []
            for i in range(len(labels)):
                model.zero_grad()
                # We need to maximize the similarity for each specific label
                score = (image_features @ text_features[i].unsqueeze(1))
                score.backward(retain_graph=True)

                # Grad-CAM logic: (Grads * Activations)
                weights = torch.mean(hook.gradients, dim=[2, 3], keepdim=True)
                cam = torch.sum(weights * hook.activations, dim=1, keepdim=True)
                cam = F.relu(cam)
                
                cam_np = cam.squeeze().cpu().detach().numpy().astype(np.float32)
                patch_heatmaps.append(cam_np)

            # 5. Visual Rendering for this Patch
            global_max = max([h.max() for h in patch_heatmaps]) + 1e-8
            num_labels = len(labels)
            
            fig, axes = plt.subplots(1, num_labels + 1, figsize=(3 * (num_labels + 1), 4))
            
            # Subplot 0: Original
            axes[0].imshow(np.array(patch))
            axes[0].set_title(f"Patch [{r},{c}]")
            axes[0].axis('off')

            # Subplots 1..N: Class Heatmaps
            for i in range(num_labels):
                h_norm = patch_heatmaps[i] / global_max
                h_resized = cv2.resize(h_norm, (patch_size, patch_size))
                
                # Apply JET colormap and overlay
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                superimposed = cv2.addWeighted(np.array(patch), 0.5, heatmap_rgb, 0.5, 0)
                
                axes[i+1].imshow(superimposed)
                axes[i+1].set_title(f"{labels[i].split()[-1]}\n{probs[i]:.1%}") # Show last word of label
                axes[i+1].axis('off')

            plt.tight_layout()
            plt.show() # Set to plt.savefig() if you want to run unattended
            # plt.close(fig) # Uncomment this if you run into memory issues

    hook.close()

# --- Execution ---
uavid_labels = [
    "aerial view of a building", 
    "aerial view of road", 
    "aerial view of a tree",
    "aerial view of low vegetation",
    "aerial view of background clutter",
    "aerial view of cars"
]

image_file = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_train\seq1\Images\file14-2.png"
analyze_all_patches_gradcam(image_file, uavid_labels)