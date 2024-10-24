import json
from re import X
from tkinter.filedialog import Directory
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.utils.multiclass import unique_labels 
from torch.utils.data import DataLoader, Dataset, random_split
from fvcore.nn import FlopCountAnalysis
import torch.nn.functional as F
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from torch.utils.data import random_split, Subset
import torch.nn.functional as F

num_classes = 1
f_size = 34 #2304 #1280

aTP , aTN, aFP, aFN = 0.0, 0.0, 0.0, 0.0
ama = 0.0
ma_list = []

valid_att_35 =     torch.tensor([False, False,  True,  True,  True,  True, False,  True, False,  False,
        False,  True, False, False,  True,  True,  False, False,  True,  True,
        False,  True, False,  True,  True, False,  True, True,  True, False,
        False, False, False, False,  True,  True,  True,  True,  True, False,
        True,  True, True,  True,  False, True,  True, True, True, False,
        True, False, False,  False, True, False, False,  True,  True,  True,
        True])

valid_att_35_low_acc = torch.tensor([True, True, True, True, True, True, True, True, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True, True, True, True, True, True, True, False])
all_true = torch.tensor([True, True, True, True, True, True, True, True, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True, True, True, True, True, True, True, True])
aug_att_num = []
for i in range(len(all_true)):
    allt = all_true.clone()
    allt[i] = False
    aug_att_num.append(allt)
    
aug_coeff = [ 1.0,  2.0,  8.0, 15.0,  4.0,  4.0,  0.1641,  0.1734,
         6.0,  6.0,  8.0, 13.0,  2.0,  2.0, 24.0,  3.0,
         0.8153,  2.0, 11.0,  0.3336,  2.0, 35.0, 11.0, 46.0,
         1.0, 28.0,  6.0, 19.0,  3.0, 59.0, 32.0,  0.9332,
        11.0,  1.0, 90.0]  
aug_coeff = [val/3 for val in aug_coeff]

aug_coeff_theo = [ 1.0,  2.0,  8.0, 15.0,  4.0,  4.0,  0.1641,  0.1734,
         6.0,  6.0,  8.0, 13.0,  2.0,  2.0, 24.0,  3.0,
         0.8153,  2.0, 11.0,  0.3336,  2.0, 35.0, 11.0, 46.0,
         1.0, 28.0,  6.0, 19.0,  3.0, 59.0, 32.0,  0.9332,
        11.0,  1.0, 89.0]  

sim_thresh = [[0.0002,0.0002], [0.0002,0.0001], [0.0005,0], [0.0008,0], [0.0002,0.0003], [0.0002,0.0002], [-0.0005,-0.0002], [-0.0003, -0.0002], [0.0004 , 0.0], [0.0003, 0.0002], [0.0004, 0.0002], [0.0006, 0.0], [0.0006, 0.0], [0.0, 0.0], [0.0006, 0.0], [0.0002, 0.0], [0.0,0.0], [0.0,0.0], [0.0002,0.0], [0.0,0.0], [0.00015, 0.00015], [0.0003,0.0003], [0.0002, 0.0001], [0.0004,0.0004], [0.0,0.0], [0.0003,0.0], [0.0002,0.0], [0.0003, 0.0], [0.0002,0.0002], [0.0007,0.0], [0.0010,0.0003], [0.0,0.0], [0.0005,0.0003], [0.0,0.0], [0.0050, 0.0070] ]

aug_coeff_exp = [ 0.0,  6.0,  4.0, 3.0,  0.0,  0.0,  0.0,  -3.0,
         2.0,  1.0,  3.0, 2.0,  1.0 ,  1.0, 4.0,  1.0,
         0.0,  0.0, 4.0,  -2.0,  1.0 , 7.0 , 5.0, 4.0,
         1.0, 5.0,  0, 6.0,  0.0, 6.0 , 8.0,  0.0,
        3.0,  0.0, 30.0]  

  # [26] error on augmentations !?!? 
valid_attrs = valid_att_35


valid_35_mask = [i for i in range(35)]
val_vneck = [34]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
device = 'cpu'

# Step 1: Load multiple JSON files and extract features and labels
data = []

#"""
directory = "annotations_BigG_no-norm/"
#directory = "annotations_ViT-amin/"
t_inputs = []
t_labels = []

for filename in os.listdir(directory):
    if filename.endswith('.json'):
        with open(os.path.join(directory, filename), 'r') as f:
            entry = json.load(f)
            #data.append(entry) 
            #filtered_labels = [entry['attributes'][i] for i in valid_attrs.nonzero(as_tuple=True)[0]]
            #data.append({'image_features': entry['image_features'], 'attributes': filtered_labels})
            feature = entry['image_features'] 
            t_inputs.append(feature)
            t_labels.append(entry['attributes'])
#t_labels = torch.tensor(t_labels, dtype=torch.float32)  
t_features = torch.tensor(t_inputs, dtype=torch.float32)

with open('splits_peta_1.json', 'r') as f:
    entry = json.load(f)
    train_indices = entry['train']
    val_indices = entry['test']


class PedestrianDataset(Dataset):
    def __init__(self, features, labels):
        #self.features = [torch.tensor(f, dtype=torch.float32) for f in features]
        #self.labels = [torch.tensor(l, dtype=torch.float32) for l in labels]
        self.features = features  # Directly use the passed tensors
        self.labels = labels
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], torch.tensor(idx).clone()

class RelationAwarenNet(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(RelationAwarenNet, self).__init__()
        self.fc1 = nn.Linear(f_size, 300)
        self.bn1 = nn.BatchNorm1d(300)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(300, 75)
        self.bn2 = nn.BatchNorm1d(75)
        self.fc3 = nn.Linear(75, num_classes)
        self.sigmoid = nn.Sigmoid()

        self.logit_scale = nn.Parameter(torch.ones(num_classes))

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        #logits = self.fc3(x)
        x = self.fc3(x)
        #x = self.sigmoid(x)
        return x
"""    
class RelationAwareNet(nn.Module):
    def __init__(self, input_size=34, num_classes=1):
        super(RelationAwareNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
"""
def calculate_mA(TP, TN, P, N, valid_attrs):
    # Calculate true positive rate (TPR) and true negative rate (TNR)
    #TPR = TP[valid_attrs] / P[valid_attrs].clamp(min=1)  # Avoid division by zero
    #TNR = TN[valid_attrs] / N[valid_attrs].clamp(min=1)
    
    TPR = TP / P.clamp(min=1)  # Avoid division by zero
    TNR = TN / N.clamp(min=1)
    
    # Calculate mean accuracy for valid attributes
    print("mA: ", (TPR+TNR)/2.0)
    print("TP: ", TPR[0]*100, "  |   TN: ", TNR[0]*100, )
    print("ALL P: ", P.clamp(min=1), "  |   All N: ", N.clamp(min=1))
    mA = torch.mean((TPR + TNR) / 2.0)

    return mA.item()  # Convert to a Python number for easier reporting

def save_metr(labels, preds):
    TP = ((preds[:] == 1) & (labels[:] == 1)).sum().item()
    TN = ((preds[:] == 0) & (labels[:] == 0)).sum().item()
    FP = ((preds[:] == 1) & (labels[:] == 0)).sum().item()
    FN = ((preds[:] == 0) & (labels[:] == 1)).sum().item()
    P = (labels[:] == 1).sum().item()  # Total Positives (actual)
    N = (labels[:] == 0).sum().item()  # Total Negatives (actual)
    
    return TP, TN, FP, FN, P, N


def compute_sample_metrics(labels, preds):
    """
    Compute the sample-wise accuracy, precision, recall, and F1 score.
    """
    

    TP = ((preds[:] == 1) & (labels[:] == 1)).sum().item()
    TN = ((preds[:] == 0) & (labels[:] == 0)).sum().item()
    FP = ((preds[:] == 1) & (labels[:] == 0)).sum().item()
    FN = ((preds[:] == 0) & (labels[:] == 1)).sum().item()
    P = (labels[:] == 1).sum().item()  # Total Positives (actual)
    N = (labels[:] == 0).sum().item()  # Total Negatives (actual)
    #print(TP,FP,TN,FN)
    #print("P:", P, "  N:",N, "  TP:",TP, "  TN:",TN)
    # Sample-wise accuracy
    sample_accuracy = (TP + TN) / (TP + FP + TN + FN + 1e-8)
    
    # Sample-wise precision
    sample_precision = TP / (TP + FP + 1e-8)
    
    # Sample-wise recall
    sample_recall = TP / (TP + FN + 1e-8)
    
    # Sample-wise F1 score
    sample_F1 = 2 * (sample_precision * sample_recall) / (sample_precision + sample_recall + 1e-8)

    return sample_accuracy, sample_precision, sample_recall, sample_F1

#####################
def generate_augmented_samples_cosine(features, labels, num_augments=5, cosine_target=0.95):
    augmented_features = []
    augmented_labels = []
    
    for feature, label in zip(features, labels):
        if label == 1:  # Only augment positive samples
            feature_norm = F.normalize(feature, dim=0)  # Normalize original feature
            
            for _ in range(num_augments):
                # Generate a random noise vector
                noise = torch.randn_like(feature)
                
                # Normalize the noise
                noise_norm = F.normalize(noise, dim=0)
                
                # Adjust the noise to match the cosine target with the original feature
                # Cosine of angle between original feature and new sample = target cosine
                noise_adjusted = cosine_target * feature_norm + (1 - cosine_target) * noise_norm
                
                # Normalize the final new sample to maintain the cosine similarity
                new_feature = noise_adjusted * torch.norm(feature)
                
                augmented_features.append(new_feature)
                augmented_labels.append(label)
    
    return augmented_features, augmented_labels
def generate_augmented_samples(features, labels, num_augments=1):
    augmented_features = []
    augmented_labels = []
    s = 1
    if (num_augments<0):
        s = 0
        num_augments *= -1
    for feature, label in zip(features, labels):
        if(label == s):
            for _ in range(num_augments):
                noise = torch.randn_like(feature) * feature * 0.05   # Adjust noise level
                new_feature = feature + noise
                augmented_features.append(new_feature)
                augmented_labels.append(label)
    #return torch.stack(augmented_features), torch.stack(augmented_labels)
    return augmented_features, augmented_labels


dataset = []

# Function to calculate Euclidean distance
def calculate_euclidean_distance(pos_features, neg_features, batch_size=1000):
    pos_features = torch.stack(pos_features)  # Shape: (num_pos, feature_dim)
    neg_features = torch.stack(neg_features)  # Shape: (num_neg, feature_dim)

    # Initialize an empty list to hold distances
    all_distances = []

    # Calculate Euclidean distances in batches
    for i in range(0, neg_features.size(0), batch_size):
        neg_batch = neg_features[i:i + batch_size]  # Shape: (min(batch_size, num_neg-i), feature_dim)
        
        # Compute squared Euclidean distance: (a - b)^2
        distances = torch.cdist(pos_features, neg_batch, p=2)  # Shape: (num_pos, min(batch_size, num_neg-i))
        all_distances.append(distances)

    # Concatenate all distance batches
    distances_tensor = torch.cat(all_distances, dim=1)  # Shape: (num_pos, num_neg)

    return distances_tensor

# Function to identify negative samples similar to positive samples using Euclidean distance
def identify_similar_negatives_euclidean(t_features, t_labels, threshold = 0.0001):
    s = 0
    if threshold < 0:
        s = 1
        threshold *= -1
    pos_features = [f for f, l in zip(t_features, t_labels) if l == 1-s]
    neg_features = [f for f, l in zip(t_features, t_labels) if l == s]
    
    pos_indices = [i for i, l in enumerate(t_labels) if l == 1-s]
    neg_indices = [i for i, l in enumerate(t_labels) if l == s]
    
    # Calculate Euclidean distance between positive and negative samples
    distances = calculate_euclidean_distance(pos_features, neg_features)  # This function calculates Euclidean distances

    # Calculate the number of negatives to remove based on the top 2%
    num_to_remove = int(threshold * len(neg_indices))
    
    # Find the top 2% most similar (smallest distance) negatives
    top_similar_neg_indices = distances.topk(num_to_remove, dim=1, largest=False).indices.flatten().tolist()

    # Map back to the original indices in t_features
    indices_to_remove = [neg_indices[i] for i in top_similar_neg_indices]

    return indices_to_remove

##############
##############

# Function to calculate cosine similarity
def calculate_cosine_similarity(pos_features, neg_features, batch_size=1000):
    pos_features = torch.stack(pos_features)  # Shape: (num_pos, feature_dim)
    neg_features = torch.stack(neg_features)  # Shape: (num_neg, feature_dim)

    # Normalize the features
    pos_features = F.normalize(pos_features, p=2, dim=1)  # Shape: (num_pos, feature_dim)
    neg_features = F.normalize(neg_features, p=2, dim=1)  # Shape: (num_neg, feature_dim)

    # Initialize an empty list to hold similarities
    all_similarities = []

    # Calculate cosine similarities in batches
    for i in range(0, neg_features.size(0), batch_size):
        neg_batch = neg_features[i:i + batch_size]  # Shape: (min(batch_size, num_neg-i), feature_dim)
        similarities = torch.mm(pos_features, neg_batch.t())  # Shape: (num_pos, min(batch_size, num_neg-i))
        all_similarities.append(similarities)

    # Concatenate all similarity batches
    similarities_tensor = torch.cat(all_similarities, dim=1)  # Shape: (num_pos, num_neg)

    return similarities_tensor

# Function to identify negative samples similar to positive samples
def identify_similar_negatives(t_features, t_labels, similarity_threshold=0.00):
    s = 0
    if similarity_threshold < 0:
        s = 1
        similarity_threshold *= -1    
    pos_features = [f for f, l in zip(t_features, t_labels) if l == 1-s]
    neg_features = [f for f, l in zip(t_features, t_labels) if l == s]
    
    pos_indices = [i for i, l in enumerate(t_labels) if l == 1-s]
    neg_indices = [i for i, l in enumerate(t_labels) if l == s]
    
    # Calculate cosine similarity (or distance) between positive and negative samples
    similarities = calculate_cosine_similarity(pos_features, neg_features)  # This function calculates cosine similarities

    # Find the top 40% most similar negatives
    num_to_remove = int(similarity_threshold * len(neg_indices))
    top_similar_neg_indices = similarities.topk(num_to_remove, dim=1).indices.flatten().tolist()

    # Map back to the original indices in t_features
    indices_to_remove = [neg_indices[i] for i in top_similar_neg_indices]

    return indices_to_remove



# Prepare dataset
dataset = []
num_att = 35
gt = [[] for _ in range(num_att)]
inp = [[] for _ in range(num_att)]
for j in range(len(t_labels)):
    for i in range(num_att):
        gt[i].append(t_labels[j][i])
        inp[i].append([])
        for iter in range(num_att):
            if iter != i:
                inp[i][j].append(t_labels[j][iter])
        inp[i][j] = torch.tensor(inp[i][j], dtype=torch.float32)
gt = torch.tensor(gt, dtype=torch.float32)        


for i in range(num_att):
    dataset.append(PedestrianDataset(inp[i], gt[i]))


train_size = len(train_indices)
val_size = len(val_indices)
total_size = train_size + val_size
# Split the dataset into train and validation sets before any modifications

# Set the seed for reproducibility


# Initialize train_loader and val_loader
train_loader = []
val_loader = []
rel_train_loader = []
rel_val_loader = []
for i in range(35):
    print("Att ",i, " input prepration...")
    # Gather the dataset for the current attribute
    

    # Create validation dataset (remains untouched)
    val_dataset = Subset(dataset[i], val_indices)
    val_loader.append(DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False))
    torch.manual_seed(42)

    # Create the train dataset before removing similar negatives
    train_dataset = Subset(dataset[i], train_indices)
    train_features = [train_dataset.dataset.features[j] for j in train_dataset.indices]
    train_labels = [train_dataset.dataset.labels[j] for j in train_dataset.indices]


    train_loader.append(DataLoader(train_dataset, batch_size=200, shuffle=True))
    """
    # Identify and remove similar negative samples
    indices_to_remove = []
    itr_ecu = []
    #if(aug_coeff[i]>1):
    indices_to_remove = identify_similar_negatives(train_features, train_labels, sim_thresh[i][0])
    itr_ecu = identify_similar_negatives_euclidean(train_features, train_labels, sim_thresh[i][1])
    indices_to_remove += itr_ecu
    # Filter the training set based on indices_to_remove
    filtered_train_features = [f for j, f in enumerate(train_features) if j not in indices_to_remove]
    print(len(filtered_train_features))
    filtered_train_labels = [l for j, l in enumerate(train_labels) if j not in indices_to_remove]
    #if (aug_coeff[i]>2):
    if not(aug_coeff_exp[i] ==0):#(aug_coeff[i]>2):
        augmented_features, augmented_labels = generate_augmented_samples(filtered_train_features, filtered_train_labels, num_augments=int(aug_coeff_exp[i]))#int(aug_coeff[i]/2))
        #augmented_features, augmented_labels = generate_augmented_samples_cosine(filtered_train_features, filtered_train_labels, num_augments=int(aug_coeff[i]))
        filtered_train_features = filtered_train_features + augmented_features
        filtered_train_labels = filtered_train_labels + augmented_labels

        filtered_train_dataset = PedestrianDataset(filtered_train_features, filtered_train_labels)
        torch.manual_seed(42)

        train_loader.append(DataLoader(filtered_train_dataset, batch_size=200, shuffle=True))            
        print(len(filtered_train_features))

    else:
        filtered_train_dataset = PedestrianDataset(filtered_train_features, filtered_train_labels)
        torch.manual_seed(42)

        train_loader.append(DataLoader(filtered_train_dataset, batch_size=200, shuffle=True))            
        print(len(filtered_train_features))
    """  


model = []
criterion = []
optimizer = []
rel_model = []
rel_criterion = []
rel_optimizer = []

for i in range(35):
    model.append(RelationAwarenNet().to(device))
    pos_weight = torch.tensor(1.0).to(device)  # Move to device
    criterion.append(nn.BCEWithLogitsLoss(pos_weight = pos_weight))
    optimizer.append(optim.Adam(model[i].parameters(), lr=0.001))




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model[0])
print(f"The model has {num_params} trainable parameters.")


def calculate_ma(labels, preds, val_proc):
    
    # True Positives, False Positives, True Negatives, False Negatives
    
    TP = ((preds[:] == 1) & (labels[:] == 1)).sum().item()
    TN = ((preds[:] == 0) & (labels[:] == 0)).sum().item()
    P = (labels[:] == 1).sum().item()  # Total Positives (actual)
    N = (labels[:] == 0).sum().item()  # Total Negatives (actual)
    #print (P, N)
    # TPR: True Positive Rate (Sensitivity), TNR: True Negative Rate (Specificity)
    TPR = TP / P if P > 0 else 0  # Handle division by zero
    TNR = TN / N if N > 0 else 0  # Handle division by zero
    #print(TP, "\n", TN, "\n\n\n")
    # Mean Accuracy for this attribute
    class_ma = (TPR + TNR) / 2
    #print(TPR, TNR)
    # Mean accuracy (mA) is the average of all class mean accuracies
    if(val_proc==3):
        print("P: ", P)
        print("N: ", N)
    
    return class_ma

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs, iterate=0):
    num_epochs = epochs
    max_val_ma_value = 0
    max_val_ma_idx = 0
    threshold = 0.5
    global aTP
    global aTN
    global aFP
    global aFN
    global ama
    global ma_list
    
    for epoch in range(epochs):

        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        train_ma = 0.0
        
        for features, labels, oi in train_loader:
            features, labels = features.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(features).squeeze(1)  # Get predictions, squeeze to match label size
            
            # Calculate loss
            #print(outputs, labels)
            loss = criterion(outputs, labels.float())  # Convert labels to float for BCEWithLogitsLoss
            
            # Backward pass and optimize
            # If still getting the error, print the gradient:
           
            loss.backward()
            
            optimizer.step()
            
            # Accumulate loss and accuracy
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            #preds = torch.round(torch.sigmoid(outputs))  # Get binary predictions (0 or 1)
            correct += (preds == labels).sum().item()
            

            train_ma += calculate_ma(labels, preds,4)
            
         

        train_loss = running_loss / len(train_loader)
        train_ma = train_ma / len(train_loader)  # Average mA over batches
        
        ####################
        # Validation Phase
        ####################
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct = 0
        val_ma = 0.0
        total_pos_confidence = 0.0  
        total_neg_confidence = 0.0
        num_pos_samples = 0         # Count of positive samples
        num_neg_samples = 0 

        with torch.no_grad():  # Disable gradient calculation during validation
            for features, labels, original_indices in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(features).squeeze(1)
                
                # Calculate loss
                loss = criterion(outputs, labels.float())
                
                val_loss += loss.item()
                
                # Calculate accuracy
                #print(outputs)
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).float()
                #preds = torch.round(torch.sigmoid(outputs))
                #print(preds)
                correct += (preds == labels).sum().item()

                # Calculate mA (mean accuracy)
                val_ma += calculate_ma(labels, preds, 4)

                # Aggregate confidence scores for positive and negative samples
                for prob, label in zip(probs, labels):
                    if label == 1:
                        total_pos_confidence += prob.item()  # Confidence for positives is directly prob
                        num_pos_samples += 1
                    else:
                        total_neg_confidence += (1 - prob.item())  # Confidence for negatives is 1 - prob
                        num_neg_samples += 1
                

        val_loss = val_loss / len(val_loader)
        val_ma = val_ma / len(val_loader)  # Average mA over batches
        if val_ma > max_val_ma_value:
            max_val_ma_value = val_ma
            max_val_ma_idx = epoch
            TP, TN, FP, FN, P, N = save_metr(labels, preds)
            file_path = "model-rel_state_full/att" + str(iterate) + ".pth"
            torch.save(model, file_path)
         
        
        acc, prec, rec, F1 = compute_sample_metrics(labels, preds)
        acc = round(acc*100,1)
        prec = round(prec*100,1)
        rec = round(rec*100,1)
        F1 = round(F1*100,1)
        # Print statistics for this epoch
        #print(f"Epoch [{epoch+1}/{num_epochs}], "
        #      f"Train Loss: {train_loss:.4f}, Train mA: {train_ma:.4f}, "
        #      f"Val Loss: {val_loss:.4f}, Val mA: {val_ma:.4f}") 
        #print("Accuracy: ",acc, " | Precision:", prec, " | Recall: ",rec, " | F1: ", F1, " \n")
        if (epoch == epochs-1):
        # Print statistics for this epoch
            print(f"Epoch @ [{max_val_ma_idx+1}/{num_epochs}], "
              #f"Train Loss: {train_loss:.4f}, Train mA: {train_ma:.4f}, "
              f"Max Val mA: {max_val_ma_value:.4f}")
            print("TPR: ",TP/P," | TNR: ", TN/N)
            print("Pos_Conf: ", round(total_pos_confidence/num_pos_samples*100,1), " | Neg_Conf: ", round(total_neg_confidence/num_neg_samples*100,1))
            aTP += TP
            aTN += TN
            aFP += FP
            aFN += FN
            ama += max_val_ma_value
            ma_list.append(round(max_val_ma_value*100,1))
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mA': max_val_ma_value ,
                'TPR': TP/P,
                'TNR': TN/N
            }
            file_path = "model-rel_state/att" + str(iterate) + ".pth"
            #torch.save(state, file_path)

################################################################

no_att = 35
#train_and_validate(model, train_loader, val_loader, criterion, optimizer, 150, validate_every=10)
for i in range(no_att):
    print("Attribute ", i , " :")
    train_and_validate(model[i].to(device), train_loader[i], val_loader[i], criterion[i].to(device), optimizer[i], epochs=30, iterate=i)
    print("\n")

acc = (aTP + aTN) / (aTP + aFP + aTN + aFN + 1e-8)
    
# Sample-wise precision
prec = aTP / (aTP + aFP + 1e-8)

# Sample-wise recall
rec = aTP / (aTP + aFN + 1e-8)

# Sample-wise F1 score
F1 = 2 * (prec * rec) / (prec + rec + 1e-8)
acc = round(acc*100,2)
prec = round(prec*100,2)
rec = round(rec*100,2)
F1 = round(F1*100,2)
ama /=  no_att
ama = round(ama*100,3)
print("list of all mAs: \n", ma_list)
print("\nFinal Results:\n\n")
print("Accuracy: ",acc, " | Precision:", prec, " | Recall: ",rec, " | F1: ", F1, " | mA: ", ama)

