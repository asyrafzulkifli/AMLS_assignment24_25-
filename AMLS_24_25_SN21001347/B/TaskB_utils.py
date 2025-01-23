import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
from torchvision import transforms
from medmnist import INFO

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
    
def Save_Model(model, name):
    model_path = os.path.join(script_dir, f"./Saved Models/{name}.pth")
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")

def Load_Model(name):
    model_path = os.path.join(script_dir, f"./Saved Models/{name}.pth")
    model = torch.load(model_path,weights_only=False)
    print(f"Model loaded from {name}.pth")
    return model

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

    model = Load_Model('Feature_Extractor')
    feature_extractor = FeatureExtractor(model).to(device)
    feature_extractor.eval()  # Set to evaluation mode

    train_loader,_,test_loader = Load_Data()

    train_features, train_labels = Features_Labels(train_loader, device)
    test_features, test_labels = Features_Labels(test_loader, device)
    print("Feature extracted")
    return train_features, test_features, train_labels, test_labels 

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
    print(info['label'])  # Outputs the label mappings