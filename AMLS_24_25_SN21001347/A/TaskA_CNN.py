import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import sys
from matplotlib import rcParams
try:
    from A.TaskA_utils import BreastMNISTDataset, set_seed, save_csv, Save_Model  # Import the custom dataset class (dataType, data_path, transform) 
except ImportError:
    # Add the parent directory to sys.path for local imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from TaskA_utils import BreastMNISTDataset, set_seed, save_csv, Save_Model  # Import the custom dataset class (dataType, data_path, transform) 

# Change font globally
rcParams['font.family'] = 'Arial'

## Loading and preprocessing data
def Load_Data(resnet=False):
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
    
    if resnet==True:
        transform = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],  # ResNet-50 normalization
                             std=[0.5]),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
    ]),
    'test_val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],  # ResNet-50 normalization
                             std=[0.5]),
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
    train_loader = DataLoader(train_data, batch_size=8, sampler=sampler, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader

## Implementing the CNN model
class TaskA_CNN(nn.Module):
    def __init__(self):
        super(TaskA_CNN, self).__init__()
        self.conv_layers = nn.Sequential( # Define the convolutional layers
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # Convolutional layer 1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1), # Convolutional layer 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Max pooling layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Convolutional layer 3
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=1), # Convolutional layer 4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential( # Define the fully connected layers
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

## Implementing Resnet18 CNN model
class TaskA_ResNet18(nn.Module):
    def __init__(self):
        super(TaskA_ResNet18, self).__init__()
        # Load the ResNet-18 model with pretrained weights
        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Change the final layer to output a single value
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.resnet18(x)

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience # Number of epochs with no improvement after which training will be stopped
        self.verbose = verbose # Print the counter
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss: # If the validation loss has decreased
            self.best_loss = val_loss # Update the best loss
            self.best_model = model.state_dict() # Save the best model
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_model(self, model):
        model.load_state_dict(self.best_model) # Load the best model

# Training the model
def Train_Model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, device, saveMode, epochs=10):
    #Create arrays to store losses and learning rates
    train_losses, val_losses, lrs, val_accuracy, epoch_list = [], [] ,[], [], []
    for epoch in range(epochs):
        epoch_list.append(epoch+1)
        model.train() # Set the model to training mode
        lrs.append(optimizer.param_groups[0]['lr'])
        running_loss = 0.0
        torch.cuda.synchronize()
        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            labels = labels.view(-1, 1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad() # Zero the gradients
            loss.backward() # Backpropagation
            optimizer.step() # Update the weights

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval() # Set the model to evaluation mode
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

        val_accuracy.append(100 * correct / total) # Calculate and store validation accuracy
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        torch.cuda.synchronize()
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")

        # Check for early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    early_stopping.restore_best_model(model) # Restore the best model in case of early stopping

    # Save the training statistics to CSV
    #save_csv('Losses', [train_losses, val_losses], ['Train Loss', 'Validation Loss'])
    #save_csv('Learning_Rates', [lrs], ['Learning Rate'])
    #save_csv('Validation Accuracy', [val_accuracy], ['Validation Accuracy'])

    # Save training plots
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, "Results"), exist_ok=True) #Create Results folder if it doesn't exist

    if saveMode:
        # Training and Validation Loss Plot
        plt.figure(figsize=(8.5 / 2.54, 8.5 / 2.54))
        plt.plot(epoch_list, train_losses, label="Train Loss")
        plt.plot(epoch_list, val_losses, label="Validation Loss")
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Losses", fontsize=10)
        plt.title("Training and Validation Loss", fontsize=10)
        plt.legend(fontsize=8)
        plt.grid()
        plt.savefig(os.path.join(script_dir, "Results", "training_validation_loss.pdf"),bbox_inches="tight")
        plt.close()

        # Learning Rate Plot
        plt.figure(figsize=(8.5 / 2.54, 8.5 / 2.54))
        plt.plot(epoch_list, [x * 1000 for x in lrs])
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel(r"Learning Rate $\times10^{-3}$", fontsize=10)
        plt.title("Learning Rate", fontsize=10)
        plt.grid()
        plt.savefig(os.path.join(script_dir, "Results", "learning_rate.pdf"),bbox_inches="tight")
        plt.close()

        # Validation Accuracy Plot
        plt.figure(figsize=(8.5 / 2.54, 8.5 / 2.54))
        plt.plot(epoch_list, val_accuracy)
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Accuracy", fontsize=10)
        plt.title("Validation Accuracy", fontsize=10)
        plt.grid()
        plt.savefig(os.path.join(script_dir, "Results", "validation_accuracy.pdf"),bbox_inches="tight")
        plt.close()

        # Save the training statistics to CSV
        save_csv('Losses', [train_losses, val_losses], ['Train Loss', 'Validation Loss'])
        save_csv('Learning_Rates', [lrs], ['Learning Rate'])
        save_csv('Validation Accuracy', [val_accuracy], ['Validation Accuracy'])

    else:
        # Training and Validation Loss Plot
        plt.figure()
        plt.plot(epoch_list, train_losses, label="Train Loss")
        plt.plot(epoch_list, val_losses, label="Validation Loss")
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Losses", fontsize=10)
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid()

        # Learning Rate Plot
        plt.figure()
        plt.plot(epoch_list, [x * 1000 for x in lrs])
        plt.xlabel("Epoch")
        plt.ylabel(r"Learning Rate $\times10^{-3}$")
        plt.title("Learning Rate")
        plt.grid()

        # Validation Accuracy Plot
        plt.figure()
        plt.plot(epoch_list, val_accuracy)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy")
        plt.grid()

# Testing the model
def Test_Model(model, test_loader, device, saveMode = True):
    all_preds = []
    all_labels = []

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.view(-1, 1)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs) > 0.5).float() # Apply sigmoid & convert to binary

            #Append predictions and labels to list
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    f1 = f1_score(all_labels, all_preds) # Calculate F1 score
    print(f"Test Accuracy: {100 * correct / total:.2f}%, F1 Score: {f1:.4f}") # Print accuracy and F1 score
    print(f"Classification Report: {classification_report(all_labels, all_preds,digits=4)}") # Print classification report

    # Confusion Matrix
    class_name = ['Malignant', 'Benign/Normal']
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    print(f"Confusion Matrix: {cm}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_name)

    if saveMode:
        # Save confusion matrix plot
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(script_dir, "Results"), exist_ok=True) #Create Results folder if it doesn't exist

        plt.figure(figsize=(8.5 / 2.54, 8.5 / 2.54))
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Normalized Confusion Matrix", fontsize=12)
        plt.savefig(os.path.join(script_dir,"Results", "confusion_matrix.pdf"),bbox_inches="tight")
        plt.close()
        # Save the confusion matrix to CSV
        save_csv('Confusion_Matrix', [all_labels, all_preds], ['True Labels', 'Predicted Labels'])

def main(saveMode = True):
    saveMode = input("Save results (Y/N) ").strip()
    if saveMode.lower() == 'y':
        saveMode = True
    elif saveMode.lower() == 'n':
        saveMode = False
    else:
        main()
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print(f"Training on {torch.cuda.get_device_name(0)}")
        torch.cuda.synchronize()

    # Step 1: Load and pre process the data
    resnet = False # Set to True to use ResNet-18 model transformations
    train_loader, val_loader, test_loader = Load_Data()

    # Step 2: Initialise the model
    if resnet:
        model = TaskA_ResNet18().to(device)
    else:
        model = TaskA_CNN().to(device)

    # Step 3: Train the model
    # Defining loss, optimiser and scheduler
    initial_lr = 0.005 # Initial learning rate
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss with Sigmoid activation
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=3)

    early_stopping = EarlyStopping(patience=10, verbose=True)
    # Train and validate the model
    Train_Model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, device, saveMode, epochs=100)

    # Step 4: Save the trained model
    if saveMode:
        Save_Model(model, "TaskA_CNN")

    # Step 4: Test and evaluate the model
    Test_Model(model, test_loader, device, saveMode)


if __name__ == "__main__":
    main(saveMode=False)

    plt.show()