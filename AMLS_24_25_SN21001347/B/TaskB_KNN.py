import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import RobustScaler
import os
import sys
try:
    from B.TaskB_utils import BloodMNISTDataset, extractFeaturesFromCNN, Train_Eval_Model
except ImportError:
    # Add the parent directory to sys.path for local imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from TaskB_utils import BloodMNISTDataset, extractFeaturesFromCNN, Train_Eval_Model

def Load_Data():
    # Load BloodMNIST data
    train_data = BloodMNISTDataset('train')
    val_data = BloodMNISTDataset('val')
    test_data = BloodMNISTDataset('test')

    x_train, y_train = train_data.images, train_data.labels.ravel()
    x_val, y_val = val_data.images, val_data.labels.ravel()
    x_test, y_test = test_data.images, test_data.labels.ravel()

    # Reshape and normalize the data
    x_train = x_train.reshape(len(x_train), -1) / 255.0
    x_test = x_test.reshape(len(x_test), -1) / 255.0

    # Scale data using RobustScaler
    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_val, x_test, y_train, y_val, y_test

def main():
    x_train, x_val, x_test, y_train, y_val, y_test = extractFeaturesFromCNN()
    #x_train, x_val, x_test, y_train, y_val, y_test = Load_Data()

    # Train KNN 
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance',metric='manhattan')

    Train_Eval_Model(knn, x_train, x_val, x_test, y_train, y_val, y_test)    
   
if __name__ == '__main__':
    main()
    plt.show()

