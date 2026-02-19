import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from tqdm import tqdm  # 1. Import tqdm

# 1. Load model and processor
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# 2. Load the full 3840x2160 image
image_path = r"C:\Users\x3non\Desktop\q3 project y2\000000.png"
image = Image.open(image_path).convert("RGB")
w, h = image.size

prompts = [
    "Building", "Road", "Static car", "Tree", 
    "Low vegetation", "Human", "Moving car", "Background clutter"
]

# 3. Setup patching parameters
patch_size = 352
stride = 176  

final_preds = torch.zeros((len(prompts), h, w))
counts = torch.zeros((1, h, w))

# Pre-calculate the steps to know the total number of patches
y_steps = list(range(0, h, stride))
x_steps = list(range(0, w, stride))
total_patches = len(y_steps) * len(x_steps)

# 4. Process the image in sliding patches with a progress bar
with tqdm(total=total_patches, desc="Processing image patches") as pbar:
    for y in y_steps:
        for x in x_steps:
            y1, x1 = min(y, h - patch_size), min(x, w - patch_size)
            y2, x2 = y1 + patch_size, x1 + patch_size
            
            patch = image.crop((x1, y1, x2, y2))
            
            inputs = processor(
                text=prompts, 
                images=[patch] * len(prompts), 
                padding=True, 
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            final_preds[:, y1:y2, x1:x2] += outputs.logits
            counts[:, y1:y2, x1:x2] += 1
            
            # Update the progress bar after each patch
            pbar.update(1)

# 5. Average the overlapping regions and apply sigmoid
final_preds = final_preds / counts
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