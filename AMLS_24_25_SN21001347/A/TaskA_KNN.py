from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from TaskA_utils import BreastMNISTDataset, extractFeaturesFromCNN
from TaskA_FE import TaskA_FE
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler
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
            block_norm='L1',     # Normalization method
            visualize=False,         # We only need features, not visualization
        )
        hog_features.append(features)
    return np.array(hog_features)

if __name__ == "__main__":
    # Step 1: Load and preprocess data
    x_train, x_test, y_train, y_test = Load_Data()

    sampler = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = sampler.fit_resample(x_train, y_train)

    x_train_resampled, x_test = extract_hog_features(x_train_resampled.reshape(-1, 28, 28)), extract_hog_features(x_test.reshape(-1, 28, 28))

    #x_train_resampled, x_test, y_train_resampled, y_test = extractFeaturesFromCNN()

    knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')
    knn.fit(x_train_resampled, y_train_resampled)

    # Step 5: Evaluate the model
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {100*accuracy:.4f}%, F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))