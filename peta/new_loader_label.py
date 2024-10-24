import os
import json
import numpy as np
from scipy.io import loadmat

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def generate_data_description(mat_file, save_dir, images_dir):
    """
    Create a dataset description file, which consists of images, labels.
    """
    # Load data from .mat file
    data = loadmat(mat_file)
    labels = data['peta'][0][0][0]
    print (data['peta'][0][0][3][0][0][0][0][0])
    print (len(data['peta'][0][0][3][0][0][0][0][1]))
    print (len(data['peta'][0][0][3][0][0][0][0][2]))
    train_idxs = []
    validation_idxs = []
    test_idxs = []
    for val in data['peta'][0][0][3][0][0][0][0][0]:
        train_idxs.append(int(val[0])-1)
    for val in data['peta'][0][0][3][0][0][0][0][1]:
        validation_idxs.append(int(val[0])-1)
    for val in data['peta'][0][0][3][0][0][0][0][2]:
        test_idxs.append(int(val[0])-1)
                

    info = {
            'train':       train_idxs,
            'validation':  validation_idxs,
            'test':        test_idxs
        }
        

        # Save JSON file for each pedestrian
    save_json(info, "splits_peta_1.json")

    # Create directories if not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return
    # Process each entry in the dataset
    for idx in range(labels.shape[0]):
      
        # Assuming the image naming convention is sequential and formatted as '00001.png', '00002.png', ...
        image_filename = f"{idx+1:05d}.png"
        image_path = os.path.join(images_dir, image_filename)

        # Create pedestrian info dictionary
        pedestrian_info = {
            'id': idx + 1,
            'attributes': labels[idx, 4:].tolist(),  # skip the first 4 columns if they are not attribute labels
            'image_path': image_path
        }
        

        # Save JSON file for each pedestrian
        #save_json(pedestrian_info, os.path.join(save_dir, f"{idx+1:05d}.json"))

if __name__ == "__main__":
    mat_file_path = 'PETA.mat'
    save_directory = 'annotations_test/'
    images_directory = 'images/'

    generate_data_description(mat_file_path, save_directory, images_directory)