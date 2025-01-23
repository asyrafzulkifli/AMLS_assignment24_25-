from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from TaskB_utils import BloodMNISTDataset, extractFeaturesFromCNN
from TaskB_FE import FE_CNN
from sklearn.preprocessing import RobustScaler
import numpy as np
import torch
import torch.nn as nn

def Load_Data_np():
    # Load BloodMNIST data
    train_data = BloodMNISTDataset('train')
    val_data = BloodMNISTDataset('val')
    test_data = BloodMNISTDataset('test')

    x_train, y_train = train_data.images, train_data.labels.ravel()
    x_val, y_val = val_data.images, val_data.labels.ravel()
    x_test, y_test = test_data.images, test_data.labels.ravel()

    # Combine train and validation sets
    x_train = np.concatenate([x_train, x_val], axis=0)
    y_train = np.concatenate([y_train, y_val], axis=0)

    # Reshape and normalize the data
    x_train = x_train.reshape(len(x_train), -1) / 255.0
    x_test = x_test.reshape(len(x_test), -1) / 255.0

    # Scale data using RobustScaler
    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test
   
if __name__ == '__main__':
    x_train, x_test, y_train, y_test = extractFeaturesFromCNN()
    #x_train, x_test, y_train, y_test = Load_Data_np()

    # Train KNN 
    TaskB_knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    TaskB_knn.fit(x_train, y_train)

    # Evaluate the model
    y_pred = TaskB_knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))