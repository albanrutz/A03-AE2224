import torch
import clip
import time

from PIL import Image
from os import listdir
from os.path import isfile, join

sequence = input("Enter which sequence from the test set: ")

mypath = "C:\\Users\\danie\\Desktop\\Delft archive\\AE2224\\archive\\uavid_test\\seq" + sequence + "\\Images"

files_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print(files_list)

# 1. Setup Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Load Model
model, preprocess = clip.load("ViT-B/32", device=device)

start_time = time.time()

for file_i in files_list:
    # 3. Load and Preprocess Image
    # REPLACE 'test_image.jpg' with the actual path to your water bottle picture
    image_path = join(mypath, file_i)
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    except FileNotFoundError:
        print(f"Error: Could not find image at {image_path}. Please check the file path.")
        exit()

    # 4. Prepare Text Prompts
    # CLIP works best when the object is used in a sentence
    text_options = ["Segment of Building in red square within drone image", "Segment of Road in red square within drone image", "Segment of Static car in red square within drone image", "Segment of Tree in red square within drone image", "Segment of Low vegetation in red square within drone image", "Segment of Human in red square within drone image", "Segment of Moving car in red square within drone image", "Segment of Background clutter in red square within drone image"]
    #text_options = ["UAV photo of Building", "UAV photo of Road", "UAV photo of Static car", "UAV photo of Tree", "UAV photo of Low vegetation", "UAV photo of Human", "UAV photo of Moving car", "UAV photo of Background clutter"]
    text_tokens = clip.tokenize(text_options).to(device)

    # 5. Calculate Features & Probabilities
    with torch.no_grad():
        # Encode image and text
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        
        # Calculate similarity (dot product) and apply softmax
        logits_per_image, logits_per_text = model(image, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # 6. Print Results

    results_dict = {}
    print(f"\n\nResults for {file_i}:")
    for i, option in enumerate(text_options):
        probability = probs[0][i] * 100
        # print(f"{option}: {probability:.2f}%")
        results_dict[option] = float(probs[0][i] * 100)

    results_dict = dict(sorted(results_dict.items(), key=lambda item: item[1], reverse=True))

    for key, value in results_dict.items():
        print(f'{key} : {value}')

end_time = time.time()

print(f'\n\nTotal time used: {end_time - start_time}')
print(f'No. of Images: {len(files_list)}')
print(f'Time used per image on avg: {(end_time - start_time)/len(files_list)}')