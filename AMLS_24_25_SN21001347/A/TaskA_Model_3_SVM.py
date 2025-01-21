from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score
from TaskA_utils import BreastMNISTDataset
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

#Step 1: Load and preprocess data
train_data = BreastMNISTDataset('train')
val_data = BreastMNISTDataset('val')
test_data = BreastMNISTDataset('test')

x_train = train_data.images.reshape(len(train_data.images), -1)/255.0
y_train = train_data.labels.ravel()
x_test = test_data.images.reshape(len(test_data.images), -1)/255.0
y_test = test_data.labels.ravel()

scaler = MinMaxScaler() #Scale data using MinMaxScaler
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

#Step 2: Train SVM
svc = SVC(gamma='scale', C=1, kernel='poly', degree=1,class_weight='balanced')
svc.fit(x_train, y_train)

#Step 3: Evaluate the model
y_pred = svc.predict(x_test)

print(f'Accuracy: {accuracy_score(y_test,y_pred):.4f} \n{classification_report(y_test, y_pred)}')