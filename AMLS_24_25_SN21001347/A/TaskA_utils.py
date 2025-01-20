import numpy as np
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random


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
        self.labels = data[type + '_labels']  # Labels (0 or 1)
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

# Function to calculate mean and std
def calculate_mean_std(dataloader):
    total_sum, total_squared_sum, num_batches = 0, 0, 0
    total_pixels = 0

    for images, _ in dataloader:
        batch_pixels = images.size(0) * images.size(2) * images.size(3)  # batch_size * H * W
        total_pixels += batch_pixels

        total_sum += torch.sum(images)  # Sum of all pixel values
        total_squared_sum += torch.sum(images ** 2)  # Sum of squared pixel values
        num_batches += 1

    mean = total_sum / total_pixels
    std = torch.sqrt((total_squared_sum / total_pixels) - (mean ** 2))
    return mean.item(), std.item()

def set_seed(seed):
    random.seed(seed)  # Python's built-in random generator
    np.random.seed(seed)  # NumPy random generator
    torch.manual_seed(seed)  # PyTorch random generator
    torch.cuda.manual_seed(seed)  # For CUDA
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility

train_data = BreastMNISTDataset('train')

unique, counts = np.unique(train_data.labels, return_counts=True)
print(unique, counts)