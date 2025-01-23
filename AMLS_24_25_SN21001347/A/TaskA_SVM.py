from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score, f1_score
from TaskA_utils import BreastMNISTDataset, extractFeaturesFromCNN
from TaskA_FE import TaskA_FE
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
from imblearn.over_sampling import SMOTE
import numpy as np
from skimage.feature import hog

def Load_Data():
    # Load BloodMNIST data
    train_data = BreastMNISTDataset('train')
    val_data = BreastMNISTDataset('val')
    test_data = BreastMNISTDataset('test')

    x_train, y_train = train_data.images, train_data.labels.ravel()
    x_val, y_val = val_data.images, val_data.labels.ravel()
    x_test, y_test = test_data.images, test_data.labels.ravel()

    # Combine train and validation sets
    x_train = np.concatenate([x_train, x_val], axis=0)
    y_train = np.concatenate([y_train, y_val], axis=0)

    # Reshape and normalize the data
    x_train = x_train.reshape(len(x_train), -1) / 255.0 # (N, 28, 28) -> (N, 784)
    x_test = x_test.reshape(len(x_test), -1) / 255.0 # (N, 28, 28) -> (N, 784)

    return x_train, x_test, y_train, y_test

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

if __name__ == "__main__":
    #Step 1: Load and preprocess data
    x_train, x_test, y_train, y_test = Load_Data()

    # Resample the training data using SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    x_train_resampled, x_test = extract_hog_features(x_train_resampled.reshape(-1, 28, 28)), extract_hog_features(x_test.reshape(-1, 28, 28)) # Extract HOG features, leave commented to use raw pixel values

    #x_train_resampled, x_test, y_train_resampled, y_test = extractFeaturesFromCNN() # Uncomment this line to use CNN features, leave commented to use raw pixel values

    #Step 2: Train SVM
    svc = SVC(gamma='scale', C=1, kernel='poly', degree=2,class_weight='balanced')
    svc.fit(x_train_resampled, y_train_resampled)

    #Step 3: Evaluate the model
    y_pred = svc.predict(x_test)

    print(f'Accuracy: {100*accuracy_score(y_test,y_pred):.4f}%, F1 Score: {f1_score(y_test, y_pred, average='weighted')} \n{classification_report(y_test, y_pred)}')