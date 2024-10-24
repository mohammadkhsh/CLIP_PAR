import os
import json
from PIL import Image, ImageOps

# Directories
image_dir = "images/"
output_dir = "pad_images/"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load JSON file with train/validation/test indices
with open("splits_peta_1.json", "r") as f:
    indices = json.load(f)

# Function to add padding to an image by 20% using center border pixels
def add_padding(image, padding_percent=0.2):
    # Calculate padding (20% of width and height)
    width, height = image.size
    padding_width = int(width * padding_percent)
    padding_height = int(height * padding_percent)

    # Get the center pixel of each border
    top_center_pixel = image.getpixel((width // 2, 0))  # Center pixel of the top border
    bottom_center_pixel = image.getpixel((width // 2, height - 1))  # Center pixel of the bottom border
    left_center_pixel = image.getpixel((0, height // 2))  # Center pixel of the left border
    right_center_pixel = image.getpixel((width - 1, height // 2))  # Center pixel of the right border

    # Add padding to each side using the center pixel of each border
    padded_image = ImageOps.expand(image, (padding_width, padding_height, padding_width, padding_height), fill=None)

    # Replace the padding on the top, bottom, left, and right with their respective center border pixels
    padded_image.paste(top_center_pixel, [0, 0, padded_image.width, padding_height])
    padded_image.paste(bottom_center_pixel, [0, padded_image.height - padding_height, padded_image.width, padded_image.height])
    padded_image.paste(left_center_pixel, [0, 0, padding_width, padded_image.height])
    padded_image.paste(right_center_pixel, [padded_image.width - padding_width, 0, padded_image.width, padded_image.height])

    return padded_image

# Process the training images with padding
for index in range(19000):#indices["train"]:
    # Convert index to image file name (e.g., 1 -> 00002.png)
    file_name = f"{index + 1:05d}.png"
    file_path = os.path.join(image_dir, file_name)

    # Load the image
    image = Image.open(file_path)

    # Add padding
    padded_image = add_padding(image)

    # Save the padded image to the output directory with the same file name
    padded_image.save(os.path.join(output_dir, file_name))

# Copy validation and test images without padding
"""
for subset in ["validation", "test"]:
    for index in indices[subset]:
        file_name = f"{index + 1:05d}.png"
        file_path = os.path.join(image_dir, file_name)

        # Copy image to the output directory as-is
        image = Image.open(file_path)
        image.save(os.path.join(output_dir, file_name))
"""
print("Padding added and images saved successfully.")