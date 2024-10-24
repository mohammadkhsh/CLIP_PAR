from PIL import Image
import os

def convert_images_to_png(source_dir, target_dir):
    """
    Convert all image files within 'archive' subfolders in the source directory to PNG format and save them in the target directory,
    preserving the original directory structure.

    Args:
    source_dir (str): The root directory where the original images are stored.
    target_dir (str): The root directory where the converted images will be stored.
    """
    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    subfolder_name = "archive"  # Specific subfolder to look for images

    # Walk through all subdirectories in the source directory
    for root, dirs, files in os.walk(source_dir):
        if subfolder_name in root:
            for file in files:
                # Check if the file is an image
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(root, file)
                    # Load the image
                    image = Image.open(image_path)

                    # Prepare the target path by maintaining the structure relative to the source directory
                    relative_path = os.path.relpath(root, source_dir)
                    target_folder = os.path.join(target_dir, relative_path)
                    
                    # Create the target directory if it does not exist
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)

                    # Save the image in PNG format
                    target_image_path = os.path.join(target_folder, f"{os.path.splitext(file)[0]}.png")
                    image.save(target_image_path, "PNG")

convert_images_to_png('PETA dataset', 'PETA_dataset_PNG')
