import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import random
import time
import csv

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load .npz file obtained from website
data_path = os.path.join(script_dir, "../Datasets/breastmnist.npz")

# Custom Dataset class for BreastMNIST
class BreastMNISTDataset(Dataset):
    def __init__(self, type, data_path=data_path, transform=None):
        # Load .npz file
        data = np.load(data_path)
        self.images = data[type + '_images']  # Images (Shape: (N, H, W))
        self.labels = data[type + '_labels'].ravel()  # Labels (0 or 1)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to PIL Image and apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label
    
def Load_Data():
    # Set seed for reproducibility
    set_seed(42)
    
    transform = {
        # Define transformations for train data (includes data augmentation)
        'train':transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
            transforms.ToTensor(),  # Convert to Tensor
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalise to [-1 1]
            transforms.RandomRotation(15),  # Rotate randomly by 15 degrees
            transforms.RandomHorizontalFlip(),  # Flip horizontally
    ]),
        # Define transformations for validation and test data
        'test_val': transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
            transforms.ToTensor(),  # Convert to Tensor
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalise to [-1 1]
    ])
    }

    #Define traning, validation and test datasets
    train_data = BreastMNISTDataset('train',transform=transform['train'])
    val_data = BreastMNISTDataset('val', transform=transform['test_val'])
    test_data = BreastMNISTDataset('test', transform=transform['test_val'])

    # Caclulate class weights for imbalanced dataset
    class_counts = [147, 399]  # Number of samples in each class in train data obtained from TaskA_utils.py
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[int(label.item())] for label in train_data.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=32, sampler=sampler, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader

def set_seed(seed):
    random.seed(seed)  # Python's built-in random generator
    np.random.seed(seed)  # NumPy random generator
    torch.manual_seed(seed)  # PyTorch random generator
    torch.cuda.manual_seed(seed)  # For CUDA
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility

def save_csv(filename,data,header=None):
    file_path = os.path.join(script_dir, f"./Results/{filename}.csv")
    with open(file_path, mode='w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for i in range(len(data)):
            writer.writerow(data[i])

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
    train_data = BreastMNISTDataset('train')
    dataset = BreastMNISTDataset('train', transform=None)

    unique, counts = np.unique(train_data.labels, return_counts=True)
    print(unique, counts)
 

    for workers in range(0, os.cpu_count() + 1, 2):
        start_time = time.time()
        dataloader = DataLoader(dataset, batch_size=32, num_workers=workers, pin_memory=True)
        for batch in dataloader:
            pass  # Simulate batch processing
        print(f"num_workers={workers}, Time taken={time.time() - start_time:.2f}s")