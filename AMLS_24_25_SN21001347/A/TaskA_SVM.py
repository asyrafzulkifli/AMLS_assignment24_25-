import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
from imblearn.over_sampling import SMOTE
import os
import sys
try:
    from A.TaskA_utils import BreastMNISTDataset, extractFeaturesFromCNN, Train_Eval_Model
except ImportError:
    # Add the parent directory to sys.path for local imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from TaskA_utils import BreastMNISTDataset, extractFeaturesFromCNN, Train_Eval_Model

def Load_Data():
    # Load BloodMNIST data
    train_data = BreastMNISTDataset('train')
    val_data = BreastMNISTDataset('val')
    test_data = BreastMNISTDataset('test')

    x_train, y_train = train_data.images, train_data.labels.ravel()
    x_val, y_val = val_data.images, val_data.labels.ravel()
    x_test, y_test = test_data.images, test_data.labels.ravel()

    # Reshape and normalize the data
    x_train = x_train.reshape(len(x_train), -1) / 255.0 # (N, 28, 28) -> (N, 784)
    x_val = x_val.reshape(len(x_val), -1) / 255.0 # (N, 28, 28) -> (N, 784)
    x_test = x_test.reshape(len(x_test), -1) / 255.0 # (N, 28, 28) -> (N, 784)

    return x_train, x_val, x_test, y_train, y_val, y_test

def extract_hog_features(images):
    hog_features = []
    for img in images:
        # Compute HOG features
        features = hog(
            img,
            orientations=12,          # Number of gradient orientations
            pixels_per_cell=(14, 14),  # Size of each cell in pixels
            cells_per_block=(2, 2),  # Number of cells per block
            block_norm='L2',     # Normalization method
            visualize=False,
        )
        hog_features.append(features)
    return np.array(hog_features)

def Tune_P(model, param_grid, x_train, y_train):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='F1')
    grid.fit(x_train, y_train)
    print("Best Parameters:", grid.best_params_)

    return grid.best_params_

def main(CV=False):
    # Load and preprocess raw data
    x_train, x_val, x_test, y_train, y_val, y_test = Load_Data()

    # Resample the training data using SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    # Extract HOG features, leave the line below commented to use raw pixel values
    x_train, x_val, x_test = extract_hog_features(x_train.reshape(-1, 28, 28)), extract_hog_features(x_val.reshape(-1, 28, 28)), extract_hog_features(x_test.reshape(-1, 28, 28)) 

    # Extract features from pre-trained CNN, leave commented to use raw pixel values/HOG features
    x_train, x_val, x_test, y_train, y_val, y_test = extractFeaturesFromCNN()
    
    if CV == True:
        print("Cross Validation is enabled.")
        # Grid search for hyperparameter tuning
        param_grid = {
            'C': [0.1, 1],
            'gamma': [1, 0.1],
            'kernel': ['poly'],
            'degree': [2, 3]
        }
        best_params = Tune_P(SVC(), param_grid, x_train, y_train)
        print("Best Parameters:", best_params)
        # Initialise SVM Classifier based on best parameters
        svc = SVC(gamma=best_params['gamma'], C=best_params['C'], kernel=best_params['kernel'], degree=best_params['degree'], class_weight='balanced')
        
    svc = SVC(gamma=1, C=0.1, kernel='poly', degree=2, class_weight='balanced')

    # Train and evaluate the model
    Train_Eval_Model(svc, x_train, x_val, x_test, y_train, y_val, y_test)

if __name__ == "__main__":
    main()
    plt.show()