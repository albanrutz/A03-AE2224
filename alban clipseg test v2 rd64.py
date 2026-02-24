import torch
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

# 3. Setup patching and batching parameters
patch_size = 352
stride = 176  
batch_size = 16 # Fits well within an 8GB RTX 4070 VRAM

final_preds = torch.zeros((len(prompts), h, w))
counts = torch.zeros((1, h, w))

# Pre-calculate patch coordinates
patch_coords = []
for y in range(0, h, stride):
    for x in range(0, w, stride):
        # Adjust boundary coordinates to ensure patches are always 352x352
        y1, x1 = min(y, h - patch_size), min(x, w - patch_size)
        y2, x2 = y1 + patch_size, x1 + patch_size
        patch_coords.append((x1, y1, x2, y2))

# 4. Process the image in batches
with tqdm(total=len(patch_coords), desc="Processing patches") as pbar:
    for i in range(0, len(patch_coords), batch_size):
        batch_coords = patch_coords[i : i + batch_size]
        batch_images = [image.crop(coords) for coords in batch_coords]
        
        # Align texts and images for the entire batch
        texts = prompts * len(batch_images)
        images_input = [img for img in batch_images for _ in range(len(prompts))]
        
        inputs = processor(
            text=texts, 
            images=images_input, 
            padding=True, 
            return_tensors="pt"
        ).to(device)
        
        # Cast image tensors to fp16 to match the model weights
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Reshape logits back to (Batch, Prompts, H, W) and move to CPU for accumulation
        logits = outputs.logits.view(len(batch_images), len(prompts), patch_size, patch_size)
        logits = logits.cpu().to(torch.float32)
        
        for idx, (x1, y1, x2, y2) in enumerate(batch_coords):
            final_preds[:, y1:y2, x1:x2] += logits[idx]
            counts[:, y1:y2, x1:x2] += 1
            
        pbar.update(len(batch_coords))

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