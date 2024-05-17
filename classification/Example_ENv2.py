#!/usr/bin/env python
# coding: utf-8

# In[1]:


log_file_path = "output_logs_ENv2_torch.txt"
image_size = (224, 224)
mode = 'rgb'
input_shape = (224, 224, 3) #depending image size and mode 
num_classes = 4
train_path = "/home/p/pakrit/MRI/brain-tumor-mri-dataset/Training"
test_path = "/home/p/pakrit/MRI/brain-tumor-mri-dataset/Testing"
checkpoint_filename = 'best_model_ENv2_torch.pth'


# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from collections import defaultdict
from efficientnet_pytorch import EfficientNet
import sys
from tqdm import tqdm
sys.stdout = open(log_file_path, "w")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU")

# In[3]:
# Define GPU device if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for training and testing data
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),                   # Resize to 224x224
    transforms.RandomHorizontalFlip(),               # Random horizontal flip
    transforms.RandomVerticalFlip(),                 # Random vertical flip
    transforms.ToTensor(),                           # Convert to tensor
    transforms.Normalize(mean=[0, 0, 0], std=[1/255, 1/255, 1/255])  # Normalize to [0, 1]
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),                   # Resize to 224x224
    transforms.ToTensor(),                           # Convert to tensor
    transforms.Normalize(mean=[0, 0, 0], std=[1/255, 1/255, 1/255])  # Normalize to [0, 1]
])

# Load training and testing datasets
train_dataset = datasets.ImageFolder(root=train_path, transform=train_transforms)
test_dataset = datasets.ImageFolder(root=test_path, transform=test_transforms)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

print("Class Names:")
print(train_dataset.classes)

print("Class Names:")
print(test_dataset.classes)


# In[ ]:


class CustomEfficientNetV2(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNetV2, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')  # You can choose other versions like 'efficientnet-b1', 'efficientnet-b2', etc.
        num_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Sequential(
            nn.Linear(num_features, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

# Initialize the model and move it to the GPU
model = CustomEfficientNetV2(num_classes=len(train_dataset.classes)).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00015)

# Define your train function here...
def train(model, train_loader, test_loader, criterion, optimizer, num_epochs=100):
    best_accuracy = 0.0
    print("Starting training loop")
    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            progress_bar.set_description(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {correct/total:.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # Evaluate the model on the test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to GPU
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")

        # Save the model if it has better accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint_filename)

# Call the train function
train(model, train_loader, test_loader, criterion, optimizer, num_epochs=100)
# In[ ]:




