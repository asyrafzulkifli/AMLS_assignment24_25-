import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from TaskA_utils import BreastMNISTDataset, calculate_mean_std, set_seed  # Import the custom dataset class (dataType, data_path, transform)
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.optim.lr_scheduler import StepLR

## Loading and preprocessing data
def Load_Data():
    # Set seed for reproducibility
    set_seed(42)

    # Define transformations for train data (includes data augmentation)
    transform_trainData = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalise using calculated parameters
        transforms.RandomRotation(15),  # Rotate randomly by 15 degrees
        transforms.RandomHorizontalFlip(),  # Flip horizontally
    ])

    # Define transformations for validation and test data
    transform_testvalData = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalise using calculated parameters
    ])

    #Define traning, validation and test datasets
    train_data = BreastMNISTDataset('train',transform=transform_trainData)
    val_data = BreastMNISTDataset('val', transform=transform_testvalData)
    test_data = BreastMNISTDataset('test', transform=transform_testvalData)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    mean, std = calculate_mean_std(train_loader)
    print(f"Train Dataset Mean: {mean:.4f}")
    print(f"Train Dataset Std: {std:.4f}")

    return train_loader, val_loader, test_loader

## Implementing the CNN model
def Implement_CNN(device='cpu'):
    # Define the CNN Model
    class BreastCancerCNN(nn.Module):
        def __init__(self):
            super(BreastCancerCNN, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)        
            )
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            x = self.conv_layers(x)
            x = self.fc_layers(x)
            return x

    model = BreastCancerCNN().to(device)

    return model

# Training the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10, device='cpu'):
    #Create arrays to store losses and learning rates
    train_losses, val_losses, lrs = [], [] ,[]
    for epoch in range(epochs):
        model.train()
        lrs.append(optimizer.param_groups[0]['lr'])
        running_loss = 0.0
        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            labels = labels.view(-1, 1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        scheduler.step()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            #Validation loop
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                labels = labels.view(-1, 1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses, lrs

# Testing the model
def test_model(model, test_loader):
    all_preds = []
    all_labels = []

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.view(-1, 1)
            outputs = model(images)
            predicted = (outputs > 0.5).float()

            #Append predictions and labels to list
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    f1 = f1_score(all_labels, all_preds)

    print(f"Test Accuracy: {100 * correct / total:.2f}%, F1 Score: {f1:.4f}")

    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))

    return all_preds, all_labels

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
train_loader, val_loader, test_loader = Load_Data()

#Initialise the model
model = Implement_CNN(device=device)

# Loss, optimiser and scheduler
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5) #Scheduler to adjust learning rate

# Train and validate the model
train_losses, val_losses, lrs = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100, device=device)

#Test model using test data
all_preds, all_labels = test_model(model, test_loader)

# Plot training and validation loss
plt.figure(1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Losses")
plt.legend()

plt.figure(2)
plt.plot(lrs)
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Step Decay Learning Rate")

cm = confusion_matrix(all_labels, all_preds, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
disp.plot(cmap=plt.cm.Blues)  # Set normalize='true'
plt.title("Normalized Confusion Matrix")
plt.show()