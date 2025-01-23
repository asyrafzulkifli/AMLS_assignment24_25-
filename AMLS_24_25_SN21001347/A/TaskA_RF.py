from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler
from skimage.feature import hog
from TaskA_utils import BreastMNISTDataset, extractFeaturesFromCNN, Train_Eval_Model
from TaskA_FE import TaskA_FE
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt

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
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(x_train, y_train)
    print("Best Parameters:", grid.best_params_)

    return grid.best_params_

if __name__ == "__main__":
    # Load and preprocess raw data
    x_train, x_val, x_test, y_train, y_val, y_test = Load_Data()

    # Resample the training data using SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    # Extract HOG features, leave the line below commented to use raw pixel values/CNN features
    x_train, x_val, x_test = extract_hog_features(x_train.reshape(-1, 28, 28)), extract_hog_features(x_val.reshape(-1, 28, 28)), extract_hog_features(x_test.reshape(-1, 28, 28)) 

    # Extract features from pre-trained CNN, leave commented to use raw pixel values/HOG features
    x_train, x_val, x_test, y_train, y_val, y_test = extractFeaturesFromCNN() 

    # Grid search for hyperparameter tuning
    param_grid = {
        'random_state': [42], # Set for reproduceability
        'class_weight': ['balanced'],
        'max_depth': [10, 20], # Tends to choose higher value
        'max_features': ['log2'],
        'min_samples_leaf': [1, 10], # Tends to choose lower value
        'min_samples_split': [2, 10], # Tends to choose higher value
        'n_estimators': [300, 400] # Tends to choose lower value
    }
    #best_params = Tune_P(RandomForestClassifier(), param_grid, x_train, y_train)
    #rf = RandomForestClassifier(**best_params)

    # Initialise SVM Classifier
    rf = RandomForestClassifier(
        random_state=42, class_weight='balanced',
        max_depth=20,
        max_features='log2',
        min_samples_leaf=1,
        min_samples_split=10,
        n_estimators=300
        )

    # Train and evaluate the model
    Train_Eval_Model(rf, x_train, x_val, x_test, y_train, y_val, y_test)
    plt.show()