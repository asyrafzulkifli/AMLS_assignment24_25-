import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler
from imblearn.over_sampling import SMOTE
import sys
import os

# Adjust imports for local execution
try:
    from A.TaskA_utils import extractFeaturesFromCNN, Train_Eval_Model, Load_Data_np
except ImportError:
    # Add the parent directory to sys.path for local imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from TaskA_utils import extractFeaturesFromCNN, Train_Eval_Model, Load_Data_np


def extract_hog_features(images):
    hog_features = []
    for img in images:
        # Compute HOG features
        features = hog(
            img,
            orientations=12,          # Number of gradient orientations
            pixels_per_cell=(14, 14),  # Size of each cell in pixels
            cells_per_block=(2, 2),  # Number of cells per block
            block_norm='L1',     # Normalization method
            visualize=False,         # We only need features, not visualization
        )
        hog_features.append(features)
    return np.array(hog_features)

def Tune_P(model, param_grid, x_train, y_train):
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(x_train, y_train)
    print("Best Parameters:", grid.best_params_)

    return grid.best_params_

def main():
    # Load and preprocess raw data
    x_train, x_val, x_test, y_train, y_val, y_test = Load_Data_np()

    # Resample the training data using SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    # Extract HOG features, leave the line below commented to use raw pixel values/CNN features
    x_train, x_val, x_test = extract_hog_features(x_train.reshape(-1, 28, 28)), extract_hog_features(x_val.reshape(-1, 28, 28)), extract_hog_features(x_test.reshape(-1, 28, 28)) 

    # Extract features from pre-trained CNN, leave commented to use raw pixel values/HOG features
    x_train, x_val, x_test, y_train, y_val, y_test = extractFeaturesFromCNN() 

    # Tune hyperparameters using GridSearchCV (Uncomment the best_params line)
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    #best_params = Tune_P(KNeighborsClassifier(), param_grid, x_train, y_train)
    #knn = KNeighborsClassifier(**best_params) # Use the best parameters found by GridSearchCV

    # Initialise SVM Classifier
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')

    # Train and evaluate the model
    Train_Eval_Model(knn, x_train, x_val, x_test, y_train, y_val, y_test)

if __name__ == "__main__":
    main()
    plt.show()

    
