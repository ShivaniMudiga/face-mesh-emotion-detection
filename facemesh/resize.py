import cv2
import os

# Paths
input_folder = "dataset/sad"      # Folder containing original images
output_folder = "dataset/sad_resized_images"   # Folder to save resized images
target_size = (256, 256)           # Desired resolution (width, height)

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    
    # Read the image
    img = cv2.imread(img_path)
    if img is not None:
        # Resize the image
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        # Convert .webp to .jpg (if needed)
        if filename.endswith(".webp"):
            filename = filename.replace(".webp", ".jpg")  # Change extension
        
        # Save resized image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, resized_img)
        print(f"Resized and saved: {output_path}")
    else:
        print(f"Error loading image: {filename}")
