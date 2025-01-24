import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import os
import sys
try:
    from B.TaskB_utils import BloodMNISTDataset, Save_Model, set_seed, save_csv
except ImportError:
    # Add the parent directory to sys.path for local imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from TaskB_utils import BloodMNISTDataset, Save_Model, set_seed, save_csv

# Change font globally
rcParams['font.family'] = 'Arial'

## Loading and preprocessing data
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

    # Caclulate class weights for imbalanced dataset
    class_counts = [852, 2181, 1085, 2026,  849,  993, 2330, 1643]  # Number of samples in each class in train data obtained from TaskA_utils.py
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[int(label.item())] for label in train_data.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

## Construct the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=8):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),     
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                    
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*6*6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    
class TaskB_CNN(nn.Module):
    def __init__(self, num_classes=8):
        super(TaskB_CNN, self).__init__()
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

class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience # Number of epochs with no improvement after which training will be stopped
        self.verbose = verbose # Print the counter
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
        model.load_state_dict(self.best_model) # Load the best model

# Training the model
def Train_Model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, device="cpu", saveMode=False, epochs=10):
    ## Create lists to store losses and learning rates
    train_losses, val_losses, lrs, val_accuracy, epoch_list = [], [], [], [], []
    for epoch in range(epochs):
        epoch_list.append(epoch+1)
        model.train()
        lrs.append(optimizer.param_groups[0]['lr'])
        running_loss = 0.0
        ## Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
                        
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        ## Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy.append(100 * correct / total)
        val_loss/=len(val_loader)
        val_losses.append(val_loss)

        ## Update the learning rate based on the validation loss
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {100 * correct / total:.2f}%, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
            ## Check for early stopping state with validation loss
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    early_stopping.restore_best_model(model) ## Restore the best model in case of early stopping
        
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
def Test_Model(model, test_loader, device="cpu", saveMode=False):
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds,digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    print(f"Confusion Matrix: \n{cm}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

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

def main(saveMode=False):
    # Ask user if they want to save the results
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
    else:
        print("Training on CPU")

    # Step 1: Load and pre process the data
    train_loader, val_loader, test_loader = Load_Data()

    # Step 2: Initialise the model
    model = TaskB_CNN(num_classes=8).to(device)

    # Step 3: Train the model
    # Defining loss, optimiser and scheduler
    initial_lr = 0.001 # Initial learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=4) #Scheduler to adjust learning rate

    ## Initialise early stopping
    early_stopping = EarlyStopping(patience=50, verbose=True)

    ## Train and validate the model
    Train_Model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, device, saveMode, epochs=50)

    if saveMode:
    # Step 4: Save the trained model
        Save_Model(model, "TaskB_CNN")

    # Step 5: Test and evaluate the model
    Test_Model(model, test_loader, device, saveMode)

if __name__ == "__main__":
    main(saveMode=True)
    plt.show()
