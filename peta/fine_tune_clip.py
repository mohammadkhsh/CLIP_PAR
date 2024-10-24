import os
import json
import torch
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim


# Define dataset class to load images and labels
class PetaDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, transform=None, limit=10):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform
        self.image_files = sorted(os.listdir(image_folder))[:limit]  # Limit to first 100 images

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        annotation_path = os.path.join(self.annotation_folder, self.image_files[idx].replace('.png', '.json'))

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations if defined
        if self.transform:
            image = self.transform(image)

        # Load the annotation and extract the label vector (first 35 elements)
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)
        label = np.array(annotation['attributes'][:35], dtype=np.float32)

        return image, torch.tensor(label)

# Define the CLIP fine-tuning model
class CLIPFineTuneModel(nn.Module):
    def __init__(self, clip_model, num_labels):
        super(CLIPFineTuneModel, self).__init__()
        self.clip_model = clip_model
        self.fc = nn.Linear(clip_model.visual.output_dim, num_labels)

    def forward(self, image):
        # Extract features from the image using CLIP's image encoder
        image_features = self.clip_model.encode_image(image)
        # Forward through the classification head
        return self.fc(image_features)

# Main function to train the model
def train_clip_finetuning(image_folder, annotation_folder, num_labels, epochs=10, batch_size=64, lr=1e-5):
    # Load the CLIP model and preprocessing
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    print (device)
    device = "cpu"
    model_name = 'ViT-bigG-14-CLIPA-336'
    clip_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='datacomp1b')
    # Prepare dataset and dataloaders
    transform = preprocess
    dataset = PetaDataset(image_folder, annotation_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize fine-tuning model
    model = CLIPFineTuneModel(clip_model, num_labels).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary classification for each of the 35 attributes
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}")

        # Print average loss for the epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(dataloader)}")

    # Save the fine-tuned model
    torch.save(model.state_dict(), "finetuned_clip_model.pth")
    print("Model saved as finetuned_clip_model.pth")

# Define parameters
image_folder = "images/"  # Path to your images
annotation_folder = "annotations_new/"  # Path to your annotation files
num_labels = 35  # Number of attributes in the PETA dataset (first 35 elements)
epochs = 10
batch_size = 2
learning_rate = 1e-5

# Start training
train_clip_finetuning(image_folder, annotation_folder, num_labels, epochs, batch_size, learning_rate)