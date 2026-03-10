import cv2
import matplotlib.pyplot as plt

# 1. Load the image using OpenCV (Loads as BGR by default)
image_path = r"C:\Users\danie\Desktop\Delft archive\AE2224\archive\uavid_val\seq16\Labels\000000.png" 
img_bgr = cv2.imread(image_path)

# Verify the image loaded successfully
if img_bgr is None:
    print(f"Error: Could not read image at {image_path}")
else:
    # 2. Convert the color space from BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 3. Display the image using Matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.axis('off')  # Hides the coordinate axes for a cleaner visualization
    plt.title("UAV Image Display")
    plt.tight_layout()
    plt.show()