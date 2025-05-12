import os
from PIL import Image

# Path to the "Train" directory
train_dir = r"Datasets/IRL_4_channel_dataset/Train/"

# Target size for resizing
target_size = (224, 224)

# Iterate through all files in the "Train" directory
for filename in os.listdir(train_dir):
    file_path = os.path.join(train_dir, filename)
    
    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        try:
            # Open the image
            with Image.open(file_path) as img:
                # Resize the image
                if img.size != (350, 350):
                    print(f"Image {file_path} has size {img.size}")
                img_resized = img.resize(target_size)
                
                # Save the resized image (overwrite the original)
                img_resized.save(file_path)
                print(f"Resized and saved: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")