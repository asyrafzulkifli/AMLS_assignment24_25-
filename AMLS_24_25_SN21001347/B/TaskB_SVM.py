import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from TaskB_utils import BloodMNISTDataset,extractFeaturesFromCNN
from TaskB_FE import FE_CNN
import matplotlib.pyplot as plt

def Load_Data():
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

def Train_Eval_SVM(x_train, x_test, y_train, y_test):
    # Train the SVM Classifier
    svc = SVC(gamma='scale', C=0.1, kernel='linear')
    svc.fit(x_train, y_train)

    # Evaluate the model on the test set
    y_pred = svc.predict(x_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()

if __name__ == "__main__":
    #x_train, x_test, y_train, y_test = Load_Data()
    x_train, x_test, y_train, y_test = extractFeaturesFromCNN()
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    Train_Eval_SVM(x_train, x_test, y_train, y_test)
    plt.show()