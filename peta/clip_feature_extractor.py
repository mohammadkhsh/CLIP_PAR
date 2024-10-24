import torch
import open_clip
from PIL import Image
import os
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import json

def progress_bar(progress, total,prefix='', fill='O'):
    iterations = 50
    length= 50

    percent = (progress / total) * 100
    filled_length = int(length * progress // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    sys.stdout.write(f"\r{prefix} |{bar}| {percent:.1f}% Complete ")

    sys.stdout.flush()
    #sys.stdout.write('\n')

device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
device = 'cpu'
model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14-CLIPA-336', pretrained='datacomp1b')
#tokenizer = open_clip.get_tokenizer('ViT-bigG-14-CLIPA-336')

model.to(device)

def open_clip_model (img_file_path):
    image = preprocess(Image.open(img_file_path)).unsqueeze(0).to(device)
    #text = tokenizer(text_list).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        #text_features = model.encode_text(text)
        #image_features /= image_features.norm(dim=-1, keepdim=True)
        #text_features /= text_features.norm(dim=-1, keepdim=True)
        #scores = (100.0 * image_features @ text_features.T)
    return image_features.tolist()[0]

def open_clip_model_for_parts(img_file_path):
    # Load the image
    image = Image.open(img_file_path)

    # Get image dimensions
    width, height = image.size
    
    # Divide the image into 3 parts (upper, middle, bottom)
    upper_part = image.crop((0, 0, width, height // 3))
    middle_part = image.crop((0, height // 3, width, 2 * height // 3))
    bottom_part = image.crop((0, 2 * height // 3, width, height))
    
    # Apply the preprocessing function to each part
    upper_part_preprocessed = preprocess(upper_part).unsqueeze(0).to(device)
    middle_part_preprocessed = preprocess(middle_part).unsqueeze(0).to(device)
    bottom_part_preprocessed = preprocess(bottom_part).unsqueeze(0).to(device)
    
    # List to hold the feature vectors
    feature_vectors = []

    with torch.no_grad():
        # Encode each part using the CLIP model's image encoder
        upper_features = model.encode_image(upper_part_preprocessed)
        middle_features = model.encode_image(middle_part_preprocessed)
        bottom_features = model.encode_image(bottom_part_preprocessed)

        # Normalize each feature vector
        upper_features /= upper_features.norm(dim=-1, keepdim=True)
        middle_features /= middle_features.norm(dim=-1, keepdim=True)
        bottom_features /= bottom_features.norm(dim=-1, keepdim=True)

        # Convert to list and append
        feature_vectors.append(upper_features.tolist()[0])
        feature_vectors.append(middle_features.tolist()[0])
        feature_vectors.append(bottom_features.tolist()[0])

    return feature_vectors


def update_json_with_image_vector(json_folder):
    # Traverse the folder containing JSON files
    file_num = 0
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            file_num += 1
            progress_bar(file_num, 19000)
            json_path = os.path.join(json_folder, filename)
            json_out = os.path.join(out_json_folder, filename)
            # Read the JSON file
            with open(json_path, 'r') as file:
                data = json.load(file)

            # Process the image and get the vector
            image_path = "pad_" + data['image_path']
            image_vector = open_clip_model(image_path)
            #image_vector = open_clip_model_for_parts(image_path)
            
            data['image_features'] = image_vector

            # Save the updated JSON file
            with open(json_out, 'w') as file:
                json.dump(data, file, indent=4)

# Example usage
json_folder = 'annotations_new/'
out_json_folder = 'annotations_pad_BigG_no-norm'
update_json_with_image_vector(json_folder)