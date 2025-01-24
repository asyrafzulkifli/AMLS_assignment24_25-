import numpy as np
import os
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score
from medmnist import INFO
import csv

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Load .npz file obtained from website
data_path = os.path.join(script_dir, "../Datasets/bloodmnist.npz")

# Custom Dataset class for BreastMNIST
class BloodMNISTDataset(Dataset):
    def __init__(self, type, data_path=data_path, transform=None):
        # Load .npz file
        data = np.load(data_path)
        self.images = data[type + '_images']
        self.labels = data[type + '_labels'].ravel()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # Convert to PIL Image and apply transformations
        if self.transform:
            image = self.transform(image)
            image = image

        return image, label

# Function to load data into tensors
def Load_Data():
    set_seed(42)  
    # Define transformations for train data (includes data augmentation)
    transform_trainData = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomRotation(15),  # Rotate randomly by 15 degrees
        transforms.RandomHorizontalFlip(),  # Flip horizontally
    ])

    # Define transformations for validation and test data
    transform_testvalData = transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL Image
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    #Define traning, validation and test datasets
    train_data = BloodMNISTDataset('train',transform=transform_trainData)
    val_data = BloodMNISTDataset('val', transform=transform_testvalData)
    test_data = BloodMNISTDataset('test', transform=transform_testvalData)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

def set_seed(seed):
    random.seed(seed)  # Python's built-in random generator
    np.random.seed(seed)  # NumPy random generator
    torch.manual_seed(seed)  # PyTorch random generator
    torch.cuda.manual_seed(seed)  # For CUDA
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility

# Function to save the results to a CSV file
def save_csv(filename,data,header=None):
    file_path = os.path.join(script_dir, f"./Results/{filename}.csv")
    with open(file_path, mode='w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for i in range(len(data)):
            writer.writerow(data[i])
    
# Function to save a trained model
def Save_Model(model, name):
    model_path = os.path.join(script_dir, f"./Saved Models/{name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")

# Function to load a pre-trained model
def Load_Model(model, name):
    model_path = os.path.join(script_dir, f"./Saved Models/{name}.pth")
    model.load_state_dict(torch.load(model_path,weights_only=True))
    print(f"Model weights loaded from {name}.pth")

class FE_CNN(nn.Module):
    def __init__(self, num_classes=8):
        super(FE_CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Function to extract features from a CNN model
def extractFeaturesFromCNN():
    # Removing the final classification layer from the CNN model
    class FeatureExtractor(nn.Module):
        def __init__(self, custom_model):
            super(FeatureExtractor, self).__init__()
            # Remove the final classification layer
            self.feature_extractor = nn.Sequential(*list(custom_model.children())[:-1])

        def forward(self, x):
            x = self.feature_extractor(x)
            return x.view(x.size(0), -1)  # Flatten feature

    def Features_Labels(data_loader, device):
        all_features = []
        all_labels = []

        # Extract features and labels
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                output = feature_extractor(images)  # Extract features
                output = output.view(output.size(0), -1)  # Flatten the output
                all_features.append(output.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # Convert to NumPy arrays
        all_features = np.vstack(all_features)
        all_labels = np.concatenate(all_labels) 

        return all_features, all_labels

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained model
    model = FE_CNN(8).to(device)
    Load_Model(model,'Feature_Extractor')
    feature_extractor = FeatureExtractor(model).to(device)
    feature_extractor.eval()  # Set to evaluation mode

    train_loader,val_loader,test_loader = Load_Data()

    train_features, train_labels = Features_Labels(train_loader, device)
    val_features, val_labels = Features_Labels(val_loader, device)
    test_features, test_labels = Features_Labels(test_loader, device)
    print("Feature extracted")
    return train_features, val_features, test_features, train_labels, val_labels, test_labels

# Function to train and evaluate classical model
def Train_Eval_Model(model, x_train, x_val, x_test, y_train, y_val, y_test):
    # Train the SVM Classifier
    model.fit(x_train, y_train)

    # Evaluate the model on the test set
    y_val_pred = model.predict(x_val)
    print(f"Validation Accuracy: {100*accuracy_score(y_val, y_val_pred):.2f}%, Precision: {precision_score(y_val, y_val_pred, average='weighted'):.4f}")

    y_pred = model.predict(x_test)
    info = INFO['bloodmnist']
    labels = info['label']
    print("Labels:", labels)
    print(f"Accuracy: {100*accuracy_score(y_test, y_pred):.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Get model name
    model_name = model.__class__.__name__
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(cm)

    # Plot and save confusion matrix
    plt.figure(figsize=(8.5 / 2.54, 8.5 / 2.54))  # Convert width to inches for matplotlib
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix", fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    # Save to PDF
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, "Results"), exist_ok=True) #Create Results folder if it doesn't exist
    plot_path = os.path.join(script_dir, "Results", f"{model_name}_confusion_matrix.pdf")
    plt.savefig(plot_path, format="pdf", bbox_inches="tight")
    print(f"Confusion matrix saved to {plot_path}")

def visualize_images(data_loader, num_images=10):
    # Get a batch of images and labels
    images, labels = next(iter(data_loader))
    
    # Ensure we don't display more images than available
    num_images = min(num_images, len(images))
    
    # Plot the images in a grid
    plt.figure(figsize=(12, 8))
    for i in range(num_images):
        plt.subplot(2, 5, i + 1)  # Adjust the grid size (e.g., 2 rows, 5 columns)
        plt.imshow(images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Unnormalize image
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_data = BloodMNISTDataset('train')

    unique, counts = np.unique(train_data.labels, return_counts=True)
    print(unique, counts)

    data = np.load(data_path)

    # Check shapes of each split
    print("Train images shape:", data['train_images'].shape)
    print("Train labels shape:", data['train_labels'].shape)
    print("Validation images shape:", data['val_images'].shape)
    print("Validation labels shape:", data['val_labels'].shape)
    print("Test images shape:", data['test_images'].shape)
    print("Test labels shape:", data['test_labels'].shape)

        # Retrieve dataset information
    info = INFO['bloodmnist']
    labels = info['label']
    class_name = [labels[f"{i}"] for i in range(len(labels))]
    print(labels)  # Outputs the label mappings

    train_loader,_,_ = Load_Data()
    visualize_images(train_loader, num_images=10)