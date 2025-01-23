import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from TaskA_utils import BreastMNISTDataset, set_seed, Save_Model  # Import the custom dataset class (dataType, data_path, transform)
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau

## Loading and preprocessing data
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

## Implementing the CNN model
class TaskA_FE(nn.Module):
    def __init__(self):
        super(TaskA_FE, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),     
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*6*6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),                
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_model(self, model):
        model.load_state_dict(self.best_model)

# Training the model
def Train_Model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, epochs=10):
    #Create arrays to store losses and learning rates
    train_losses, val_losses, lrs = [], [] ,[]
    for epoch in range(epochs):
        model.train()
        lrs.append(optimizer.param_groups[0]['lr'])
        running_loss = 0.0
        torch.cuda.synchronize()
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

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        torch.cuda.synchronize()
        with torch.no_grad():
            #Validation loop
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                labels = labels.view(-1, 1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        torch.cuda.synchronize()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    early_stopping.restore_best_model(model)

    # Plot training and validation loss
    plt.figure(1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.legend()
    plt.title("Training and Validation Loss")

    plt.figure(2)
    plt.plot(lrs)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Step Decay Learning Rate")

# Testing the model
def Test_Model(model, test_loader):
    all_preds = []
    all_labels = []

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.view(-1, 1)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float()

            #Append predictions and labels to list
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    f1 = f1_score(all_labels, all_preds)
    print(f"Test Accuracy: {100 * correct / total:.2f}%, F1 Score: {f1:.4f}")

    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))

    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix")

if __name__ == "__main__":
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"Training on {torch.cuda.get_device_name(0)}")
        torch.cuda.synchronize()

    # Step 1: Load and pre process the data
    resnet = False # Set to True to use ResNet-18 model transformations
    train_loader, val_loader, test_loader = Load_Data()

    # Step 2: Initialise the model
    model = TaskA_FE().to(device)

    # Step 3: Train the model
    # Defining loss, optimiser and scheduler
    initial_lr = 0.001 # Initial learning rate
    class_imbalance_ratio = 19/7
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_imbalance_ratio]).to(device))  # Binary Cross-Entropy Loss with Sigmoid activation
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=4)

    early_stopping = EarlyStopping(patience=10, verbose=True)
    # Train and validate the model
    Train_Model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, epochs=100)

    # Step 4: Save the trained model
    Save_Model(model, "Feature_Extractor")

    # Step 4: Test and evaluate the model
    Test_Model(model, test_loader)

    plt.show()