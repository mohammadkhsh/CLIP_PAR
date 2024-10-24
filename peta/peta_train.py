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


num_classes = 35
f_size = 1280


def apply_smote(features, labels):
    smote = SMOTE()
    features_resampled, labels_resampled = smote.fit_resample(features, labels)
    return features_resampled, labels_resampled

def random_feature_modification(features, noise_scale=0.05):
    noise = torch.randn(features.size()) * noise_scale
    new_features = features + noise
    return new_features



valid_att_less05 = torch.tensor([False, False,  True,  True,  True,  True, False,  True, False,  True,
        False,  True, False, False,  True,  True,  True, False,  True,  True,
        False,  True, False,  True,  True, False,  True, False,  True, False,
         True, False, False, False,  True,  True,  True,  True,  True, False,
        False,  True, False,  True,  True, False,  True, False, False, False,
        False, False, False,  True, False, False, False,  True,  True,  True,
        False])

valid_att_less10 = torch.tensor([False, False,  True,  True,  True, False, False,  True, False, False,
        False,  True, False, False,  True,  True,  True, False,  True,  True,
        False,  True, False, False,  True, False,  True, False,  True, False,
         True, False, False, False,  True,  True, False,  True,  True, False,
        False, False, False,  True,  True, False,  True, False, False, False,
        False, False, False, False, False, False, False,  True, False,  True,
        False])

valid_att_35 =     torch.tensor([False, False,  True,  True,  True,  True, False,  True, False,  False,
        False,  True, False, False,  True,  True,  False, False,  True,  True,
        False,  True, False,  True,  True, False,  True, True,  True, False,
        False, False, False, False,  True,  True,  True,  True,  True, False,
        True,  True, True,  True,  False, True,  True, True, True, False,
        True, False, False,  False, True, False, False,  True,  True,  True,
        True])

valid_attrs = valid_att_35


valid_att_35_less10 = torch.tensor([True, True, True, False, True, True, True, True, True, True, True, False,
                                     True, True, False, True, True, True, False, True, True, False, False, False, True, False, True, False, False,
                                     False, False, True, False, True, False])

valid_att_35_less5 = torch.tensor([True, True, True, True, True, True, True, True, True, True, True, True,
                                     True, True, False, True, True, True, True, True, True, True, True, False, True, False, True, False, False,
                                     False, False, True, True, True, False])

valid_att_35_less5 = torch.tensor([True, True, True, True, True, True, True, True, True, True, True, True,
                                     True, True, False, True, True, True, True, True, True, True, True, False, True, False, True, False, False,
                                     False, False, True, True, True, False])

valid_att_35_less25 = torch.tensor([True, True, False, False, False, False, True, True, False, False, False, False,
                                     True, True, False, True, True, True, False, True, True, False, False, False, True, False, False, False, False,
                                     False, False, True, False, True, False])

valid_att_35_low_acc1 = torch.tensor([True, True, True, True, True, True, True, True, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True, True, False, True, True, True, True, False])
valid_att_35_low_acc = torch.tensor([True, True, True, True, True, True, True, True, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True, True, True, True, True, True, True, True, True,
                                 True, True, True, True, True, True, True, False])

valid_35_mask = [i for i in range(35)]
val_vneck = [34]


# Define the function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.viridis):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(y_true)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="black" if cm[i, j] > thresh else "white", fontsize=16)
    fig.tight_layout()
    return ax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
device = 'cpu'

# Step 1: Load multiple JSON files and extract features and labels
data = []

#"""
directory = "annotations_BigG_no-norm/"
t_features = []
t_labels = []
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        with open(os.path.join(directory, filename), 'r') as f:
            entry = json.load(f)
            #data.append(entry) 
            #filtered_labels = [entry['attributes'][i] for i in valid_attrs.nonzero(as_tuple=True)[0]]
            #data.append({'image_features': entry['image_features'], 'attributes': filtered_labels})
            feature = torch.tensor(entry['image_features'], dtype=torch.float32)
            
            label = torch.tensor([entry['attributes'][i] for i in valid_35_mask], dtype=torch.float32)
            t_features.append(feature)
            
            t_labels.append(label)

           
#img_features = [entry['image_features'] for entry in data]
#labels = [entry['attributes'] for entry in data]
#file_name = [entry["image_path"] for entry in data]

number_of_features = len(t_features[0]) # default should be 1280  bigG should be 1152
print(number_of_features)

class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention_fc = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # Apply a fully connected layer and use sigmoid to get weights between 0 and 1
        weights = torch.sigmoid(self.attention_fc(x))
        # Apply the attention weights
        attended = x * weights
        return attended, weights

class PedestrianDataset(Dataset):
    def __init__(self, features, labels):
        #self.features = [torch.tensor(f, dtype=torch.float32) for f in features]
        #self.labels = [torch.tensor(l, dtype=torch.float32) for l in labels]
        self.features = features  # Directly use the passed tensors
        self.labels = labels
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class PedestrianNet(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(PedestrianNet, self).__init__()
        self.fc1 = nn.Linear(f_size, 700)
        self.bn1 = nn.BatchNorm1d(700)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(700, 175)
        self.bn2 = nn.BatchNorm1d(175)
        self.fc3 = nn.Linear(175, num_classes)
        self.sigmoid = nn.Sigmoid()

        self.logit_scale = nn.Parameter(torch.ones(num_classes))

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        #x = self.sigmoid(x)
        return x
        #logits = self.fc3(x)
        #scaled_logits = logits * self.logit_scale
        #return scaled_logits





class CustomBCELoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, logits, labels):
        # Sigmoid activation to convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Calculate custom weighted binary cross entropy
        loss = -(self.model.loss_weights * (labels * torch.log(probs + 1e-6) + (1 - labels) * torch.log(1 - probs + 1e-6))).mean()
        return loss

def calculate_mA(TP, TN, P, N):
    # Calculate true positive rate (TPR) and true negative rate (TNR)

    TPR = TP / P.clamp(min=1)  # Avoid division by zero
    TNR = TN / N.clamp(min=1)
    
    # Calculate mean accuracy for valid attributes
    print("mA: ", (TPR+TNR)/2.0)
    print(TPR[34],TNR[34])

    mA = torch.mean((TPR + TNR) / 2.0)

    return mA.item()  # Convert to a Python number for easier reporting

def compute_sample_metrics(predictions, labels):
    """
    Compute the sample-wise accuracy, precision, recall, and F1 score.
    """
    # Convert to boolean tensors for logical operations

    predictions_bool = predictions.bool()
    labels_bool = labels.bool()


    #valid_predictions = predictions_bool[valid_attrs]
    #valid_labels = labels_bool[valid_attrs]
    valid_predictions = predictions_bool
    valid_labels = labels_bool

    # True Positives, False Positives, True Negatives, False Negatives
    TP = (valid_predictions & valid_labels).sum().float()
    FP = (valid_predictions & ~valid_labels).sum().float()
    TN = (~valid_predictions & ~valid_labels).sum().float()
    FN = (~valid_predictions & valid_labels).sum().float()
    print(TP,FP,TN,FN)
    
    # Sample-wise accuracy
    sample_accuracy = (TP + TN) / (TP + FP + TN + FN + 1e-8)
    
    # Sample-wise precision
    sample_precision = TP / (TP + FP + 1e-8)
    
    # Sample-wise recall
    sample_recall = TP / (TP + FN + 1e-8)
    
    # Sample-wise F1 score
    sample_F1 = 2 * (sample_precision * sample_recall) / (sample_precision + sample_recall + 1e-8)

    return sample_accuracy, sample_precision, sample_recall, sample_F1

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate probability that each pixel belongs to class 1
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Here we compute p_t
        pt = torch.exp(-BCE_loss)  # pt is the probability of being classified as the true class
        
        # Focal loss calculation
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


#####################
def generate_augmented_samples(features, labels, less10_mask, num_augments=5):
    augmented_features = []
    augmented_labels = []
    for feature, label in zip(features, labels):
        if any(label[less10_mask]):
            for _ in range(num_augments):
                noise = torch.randn_like(feature) * feature * 0.3   # Adjust noise level
                new_feature = feature + noise
                augmented_features.append(new_feature)
                augmented_labels.append(label)
    return torch.stack(augmented_features), torch.stack(augmented_labels)

def generate_combined_samples(features, labels, threshold_mask, num_combinations=2):
    """
    Generate new samples by averaging the features and combining labels with logical OR
    for samples that have at least one under-represented attribute.

    Args:
    features (Tensor): The original features.
    labels (Tensor): The original labels.
    threshold_mask (Tensor): A mask indicating under-represented attributes (True if under-represented).
    num_combinations (int): Number of new samples to generate from each eligible original sample.

    Returns:
    Tuple[Tensor, Tensor]: A tuple containing the augmented features and labels.
    """
    augmented_features = []
    augmented_labels = []
    
    # Identifying samples with at least one under-represented attribute
    under_rep_samples_mask = (labels[:, ~threshold_mask]).sum(1) > 0
    
    # Filtering eligible samples
    eligible_features = features[under_rep_samples_mask]
    eligible_labels = labels[under_rep_samples_mask]

    # Generate combinations
    for _ in range(num_combinations):
        indices = torch.randperm(len(eligible_features))[:2]  # Randomly pick two samples to combine
        new_feature = torch.mean(eligible_features[indices], dim=0)
        new_label = torch.logical_or(eligible_labels[indices[0]], eligible_labels[indices[1]])
        
        augmented_features.append(new_feature)
        augmented_labels.append(new_label)

    return torch.stack(augmented_features), torch.stack(augmented_labels)



#"""
# Assuming t_features and t_labels are tensors of all data
dataset = PedestrianDataset(t_features, t_labels)

# Split into training and validation sets
train_size = int(0.6 * len(dataset))
val_size = len(dataset) - train_size
torch.manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Extract features and labels from train_dataset
train_features = torch.stack([train_dataset.dataset.features[i] for i in train_dataset.indices])
train_labels = torch.stack([train_dataset.dataset.labels[i] for i in train_dataset.indices])

# Determine which attributes have less than 10% positive rate
less10_mask = ~valid_att_35_low_acc

# Generate augmented datas
augmented_features, augmented_labels = generate_augmented_samples(train_features, train_labels, less10_mask, num_augments=60)
#combined_features, combined_labels = generate_combined_samples(train_features, train_labels, less10_mask, 400)

# Create an augmented dataset
#combined_dataset = PedestrianDataset(combined_features, combined_labels) #augmented_dataset = PedestrianDataset(augmented_features, augmented_labels)
augmented_dataset = PedestrianDataset(augmented_features, augmented_labels)
# Concatenate the original train dataset with the augmented dataset
final_train_dataset =  ConcatDataset([train_dataset, augmented_dataset])
#final_train_dataset =  ConcatDataset([final_train_dataset, combined_dataset])
###

train_loader = DataLoader(final_train_dataset, batch_size=100, shuffle=True)
#train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
#batch_size=len(val_dataset),

for _, labels in train_loader:
    P = labels.sum(dim=0)
    N = (~labels.bool()).sum(dim=0)
print(P/N)

print("Training samples: ", len(train_loader)*100)
print("Validation samples: ", len(val_loader))
#"""
#train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True)
#val_loader = DataLoader(val_dataset, shuffle=False)

weights = torch.ones(num_classes)  # Default weight for all attributes
higher_weight = 1.0  # You can adjust this value as necessary
#weights[~valid_att_35_less10] = higher_weight  # Assign higher weights to less frequent attributes
#weights = weights.to(device) # Move weights to the correct device

# Assuming the default threshold is 0.5 for all attributes
default_threshold = 0.5
lower_threshold = 0.3  # Lower threshold for attributes with less than 5% positive rate

# Create a tensor of thresholds where less frequent attributes have a lower threshold
#thresholds = torch.full((num_classes,), default_threshold, device=device)
#thresholds[~valid_att_35_less10] = lower_threshold  # Apply lower threshold to less frequent attributes

#loss_function = CustomLoss(valid_att_35_less10).to(device)
#loss_function = RecallLoss(valid_att_35_less10).to(device)
#loss_function = F1Loss().to(device)

#model = PedestrianNet().to(device)
#input_size = 3456#1152  # Number of input features
hidden_size = 256  # Number of features in the hidden state
num_classes = 35   # Total number of classes
model = PedestrianNet().to(device)

criterion = nn.BCEWithLogitsLoss(weight=weights)
#criterion = FocalLoss(alpha=0.75, gamma=1.2)

optimizer = optim.Adam(model.parameters(), lr=0.001)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_parameters(model)
print(f"The model has {num_params} trainable parameters.")


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs, validate_every=10):
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_TP = 0
        train_TN = 0
        train_P = 0
        train_N = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            """
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            """
        #train_loss /= len(train_loader)
        #train_mA = compute_mean_accuracy(train_TP, train_TN, train_P, train_N)

        
        model.eval()
        val_loss = 0.0
        sum_accuracy, sum_precision, sum_recall, sum_F1 = 0.0, 0.0, 0.0, 0.0
        total_samples = 0

        # Initialize confusion matrix components for all attributes
        TP = torch.zeros(num_classes, device=device)
        TN = torch.zeros(num_classes, device=device)
        P = torch.zeros(num_classes, device=device)
        N = torch.zeros(num_classes, device=device)
        FP = torch.zeros(num_classes, device=device)
        FN = torch.zeros(num_classes, device=device)
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predictions = torch.sigmoid(outputs) > 0.5
                
                acc, prec, rec, f1 = compute_sample_metrics(predictions[0], labels[0])
                sum_accuracy += acc
                sum_precision += prec
                sum_recall += rec
                sum_F1 += f1
                total_samples += 1

                # Update confusion matrix components
                TP += (predictions & labels.bool()).sum(dim=0)
                TN += ((~predictions) & (~labels.bool())).sum(dim=0)
                P += labels.sum(dim=0)
                N += (~labels.bool()).sum(dim=0)
                FP += (predictions & ~labels.bool()).sum(dim=0)
                FN += (~predictions & labels.bool()).sum(dim=0)
                inputs_flop = inputs   
               
            
        val_loss /= len(val_loader)
        overall_acc = (TP + TN) / (TP + TN + FP + FN)
        print ("O-acc: ", overall_acc)
        
        # Calculate positive rates and determine valid attributes
        positive_rates = P / (P + N)
        ###print("perc. pos: ",positive_rates*100)
        more05att = positive_rates >= 0.05
       
        #print("Pos. rate: ",positive_rates*100)

        #TPR = TP[valid_attrs]
        #FPR = FP[valid_attrs]
        #TNR = TN[valid_attrs]
        #FNR = FN[valid_attrs]
        #accuracy, precision, recall, F1 = compute_metrics(TPR, FPR, TNR, FNR)
      

        avg_accuracy = sum_accuracy / total_samples
        avg_precision = sum_precision / total_samples
        avg_recall = sum_recall / total_samples
        avg_F1 = sum_F1 / total_samples
    

        # Compute mean accuracy for the validation set, considering only valid attributes
        val_mA = calculate_mA(TP, TN, P, N)*100
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}, Validation mA: {val_mA:.2f}')
        accuracy, precision, recall, F1 = avg_accuracy*100, avg_precision*100, avg_recall*100, avg_F1*100
        print(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {F1:.2f}')
        mFive = (val_mA + accuracy + precision + recall + F1)/5
        print(f'mFive: {mFive:.2f}')
        print()
    #print(valid_attrs)
    for inputs, labels in val_loader:
        flop_count = FlopCountAnalysis(model, inputs_flop)
        print(f"GFLOPs: {flop_count.total() / 1e9}")  # Convert FLOPs to GFLOPs 
        break
    print(total_samples)
    """
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss', linestyle='--')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy', linestyle='--')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('plot_3L_D025_lr0005.png')
    


    # Assuming 'val_loader' is your DataLoader for the validation set
    model.eval()
    examples = enumerate(val_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    # Get predictions
    with torch.no_grad():
        example_data = example_data.to(device)
        output = model(example_data)

    # Applying sigmoid since our output has logits, threshold at 0.5 for binary classification
    predicted = output.sigmoid() > 0.5
    predicted = predicted.cpu()

    # Select a few examples to display
    for i in range(6):
        img_features = example_data[i].cpu()
        img_label = example_targets[i]
        img_prediction = predicted[i]
        print(f"Example {i+1}:")
        print(f"True Labels: {img_label}")
        print(f"Predicted Labels: {img_prediction}")
        print("\n---\n")

    """    

#train_and_validate(model, train_loader, val_loader, criterion, optimizer, 150, validate_every=10)
train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=40)

