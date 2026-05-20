print("Script started")

import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import numpy as np 
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader 

from app.emotion_model import SimpleEmotionModel 

# Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

print("Loading dataset...")

# Train Dataset 
train_dataset = datasets.ImageFolder(
    "data/fer2013/train",
    transform=transform

)

# Test Dataset
test_dataset = datasets.ImageFolder(
    "data/fer2013/test",
    transform=transform
)


# data Loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# Class Names 
classes = train_dataset.classes

print("Classes:", classes)

# Model 
model = SimpleEmotionModel().to(device)

# Loss + Optimizer 
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Validation Function

def  evaluate_model(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy

# Confusion Matrix Function

def generate_confusion_matrix(model, loader, classes):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8,6))

    plt.imshow(cm, interpolation='nearest')

    plt.title("Confusion Matrix")

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    plt.show()     

# Training loop


print("Training started...")
# Training 
epochs = 5

for epoch in range(epochs):

    model.train() 

    running_loss = 0.0 

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device) 

        optimizer.zero_grad()

        outputs = model(images)
        
        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss:.4f}")
    accuracy = evaluate_model(model, test_loader)

    print(f"Validation Accuracy: {accuracy:.2f}%")


# save model 
torch.save(model.state_dict(), "models/emotion_model/emotion_model.pth")

print("\nModel training complete and saved.")

generate_confusion_matrix(model, test_loader, classes)
