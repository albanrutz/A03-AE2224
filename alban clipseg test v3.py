import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from tqdm import tqdm

device = "cuda"

# 1. Model and Token
hf_token = ""

processor = CLIPSegProcessor.from_pretrained(
    "CIDAS/clipseg-rd64-refined", 
    token=hf_token
)
model = CLIPSegForImageSegmentation.from_pretrained(
    "CIDAS/clipseg-rd64-refined", 
    torch_dtype=torch.float16,
    token=hf_token
).to(device)

# 2. Load the full 3840x2160 image
image_path = r"C:\Users\x3non\Desktop\q3 project y2\000000.png"
image = Image.open(image_path).convert("RGB")
w, h = image.size

prompts = [
    "Building", "Road", "Static car", "Tree", 
    "Low vegetation", "Human", "Moving car", "Background clutter"
]

# 3. Setup multi-scale and batching parameters
scales = [352, 176, 88]
batch_size = 16 # Fits well within 8GB VRAM (CLIPSeg always processes at 352x352 internally)

final_preds = torch.zeros((len(prompts), h, w), dtype=torch.float32)
weight_sums = torch.zeros((len(prompts), h, w), dtype=torch.float32)

# 4. Process the image across multiple scales
for patch_size in scales:
    stride = patch_size // 2  # Maintains 1/2 overlap at all scales
    
    # Pre-calculate patch coordinates
    patch_coords = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, x1 = min(y, h - patch_size), min(x, w - patch_size)
            y2, x2 = y1 + patch_size, x1 + patch_size
            patch_coords.append((x1, y1, x2, y2))
            
    # Deduplicate coordinates (min() boundary clamping can create duplicates at image edges)
    patch_coords = list(dict.fromkeys(patch_coords))
    
    with tqdm(total=len(patch_coords), desc=f"Processing {patch_size}px patches") as pbar:
        for i in range(0, len(patch_coords), batch_size):
            batch_coords_sub = patch_coords[i : i + batch_size]
            batch_images = [image.crop(coords) for coords in batch_coords_sub]
            
            texts = prompts * len(batch_images)
            images_input = [img for img in batch_images for _ in range(len(prompts))]
            
            # The processor automatically resizes inputs to 352x352 for the model
            inputs = processor(
                text=texts, 
                images=images_input, 
                padding=True, 
                return_tensors="pt"
            ).to(device)
            
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            # Logits are output at the internal 352x352 resolution
            logits = outputs.logits.view(len(batch_images), len(prompts), 352, 352)
            logits = logits.cpu().to(torch.float32)
            
            # Downscale logits back to the true patch size for correct spatial accumulation
            if patch_size != 352:
                logits = F.interpolate(
                    logits, 
                    size=(patch_size, patch_size), 
                    mode="bilinear", 
                    align_corners=False
                )
            
            # Confidence weighting: Distance from the 0 decision boundary indicates certainty
            weights = torch.abs(logits) + 1e-5 
            
            # Accumulate predictions and weights
            for idx, (x1, y1, x2, y2) in enumerate(batch_coords_sub):
                final_preds[:, y1:y2, x1:x2] += logits[idx] * weights[idx]
                weight_sums[:, y1:y2, x1:x2] += weights[idx]
                
            pbar.update(len(batch_coords_sub))

# 5. Calculate the confidence-weighted average and apply sigmoid
final_preds = final_preds / weight_sums
final_probs = torch.sigmoid(final_preds)

# 6. Visualize the overlays
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for i, (prompt, prob) in enumerate(zip(prompts, final_probs)):
    row, col = divmod(i, 4)
    ax[row, col].imshow(image)
    ax[row, col].imshow(prob.numpy(), cmap='jet', alpha=0.5)
    ax[row, col].set_title(prompt)
    ax[row, col].axis('off')

plt.tight_layout()
plt.show()