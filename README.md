# Pedestrian Attribute Recognition with CLIP on PETA Dataset

![PETA Dataset](pie_peta.png)

This project aims to achieve **pedestrian attribute recognition** on the **PETA dataset** using a **CLIP-based image encoder** as the backbone. The dataset contains 35 selected attributes, and the challenge lies in handling **class imbalance**—where some attributes have significantly fewer positive samples—resulting in lower recall and decreased **mean accuracy (mA)**. The goal is to surpass the current state-of-the-art mA of **89.89%**. The current best mA achieved is **88.3%**.

The **mA (mean accuracy)** is chosen as the key metric because it provides a balanced evaluation of the model’s ability to correctly classify both positive and negative cases. Given the class imbalance in certain attributes, a high mA indicates that the model performs well in handling these cases on average.

## Project Structure

- **PETA.mat**  
  Contains the attribute data and labels from the PETA dataset.

- **clip_feature_extractor.py**  
  Extracts image features using the ViT-bigG-14-CLIPA-336 encoder, which gives us the 1280-sized feature vector for each image and saves it in a json file with the path to the orignial image.

- **fine_tune_clip.py**  
  Code to fine-tune the CLIP model for pedestrian attribute classification on the PETA dataset.

- **image_format_converter.py**  
  Converts images into the required format for training and processing.

- **new_loader_label.py**  
  Handles loading and preprocessing of the dataset, including the extraction of attribute labels for training/val/test.

- **padder.py**  
  Adds padding to the training images, enlarging them by 20% while maintaining the border pixel values to preserve image quality.

- **peta_relation-aware.py**  
  Implements a relation-aware model, which incorporates relationships between 34 labels to predict the 35th labels. This model can then be used to improve the final accuracy by combining to the pipeline.

- **peta_single_train.py**  
  Trains a model to classify individual attributes separately, handling one attribute classification at a time. Data preprocessing, such as augmentation and removal of similar samples with opposite labels are done here.

- **peta_train.py**  
  The main training script that trains the model on all 35 attributes.

## Future Development
This project is currently **under development**. The goal is to continue improving the mA and surpass the state-of-the-art results. The research and code are **fully confidential**.
