# import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score
from TaskA_utils import BreastMNISTDataset
from sklearn.preprocessing import MinMaxScaler

train_data = BreastMNISTDataset('train')
val_data = BreastMNISTDataset('val')
test_data = BreastMNISTDataset('test')

x_train = train_data.images.reshape(len(train_data.images), -1)/255.0
y_train = train_data.labels.ravel()
x_test = test_data.images.reshape(len(test_data.images), -1)/255.0
y_test = test_data.labels.ravel()

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Initialise and train model
svc = SVC(gamma=0.001, C=1, kernel='linear')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

print(f'Accuracy: {accuracy_score(y_test,y_pred):.4f} \n{classification_report(y_test, y_pred)}')

print(f"Mean of the training data: {np.mean(x_train):.4f}, Standard deviation of the training data: {np.std(x_train):.4f}")