import numpy as np
import os
from torch.utils.data import Dataset
import numpy as np

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load .npz file obtained from website
data_path = os.path.join(script_dir, "../Datasets/breastmnist.npz")

# Custom Dataset class for BreastMNIST
class BreastMNISTDataset(Dataset):
    def __init__(self, type, data_path=data_path, transform=None):
        # Load .npz file
        data = np.load(data_path)
        self.images = data[type + '_images']  # Images (assumed shape: (N, H, W))
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