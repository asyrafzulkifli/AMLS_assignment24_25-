import numpy as np
import os
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from torch.utils.data import DataLoader
import random

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

def set_seed(seed):
    random.seed(seed)  # Python's built-in random generator
    np.random.seed(seed)  # NumPy random generator
    torch.manual_seed(seed)  # PyTorch random generator
    torch.cuda.manual_seed(seed)  # For CUDA
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility
    
def Save_Model(model, name):
    model_path = os.path.join(script_dir, f"./Saved Models/{name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def Load_Model(model, name):
    model_path = os.path.join(script_dir, f"./Saved Models/{name}.pth")
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.eval()
    return model

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