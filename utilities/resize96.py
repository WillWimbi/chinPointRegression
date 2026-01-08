import os
import cv2

rawImgs = "./rawImgs" #from another folder, obselete now
Imgs = "./Imgs" #from another folder, obselete now

def resize_and_rename_images(input_folder, output_folder, new_size=(96, 96)):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List all files in the input directory
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Filter only image files
    images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
    
    # Process each image
    for i, file in enumerate(images):
        img_path = os.path.join(input_folder, file)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Skipping {file} (not a valid image)")
            continue
        
        # Resize the image
        resized_image = cv2.resize(image, new_size)
        
        # Construct new filename
        new_filename = f"{i + 1:03d}.png"  # Naming with leading zeros and .png extension
        output_path = os.path.join(output_folder, new_filename)
        
        # Save the resized image
        cv2.imwrite(output_path, resized_image)
        
        print(f"Processed {file} -> {new_filename}")

# Call the function
resize_and_rename_images(rawImgs, Imgs)

# import torch
# import torch.nn as nn
# r = torch.rand(34, 13,90)
# print(r.shape)
# e = r.flatten(start_dim=1, end_dim=-1)
# print(e.shape)

# x = torch.randn(1024,1,1)
# print("Original shape:", x.shape)
# flatten = nn.Flatten(0)
# out = flatten(x)
# print("Default flatten shape:", out.shape)
# # Output: Default flatten shape: torch.Size([2, 60])