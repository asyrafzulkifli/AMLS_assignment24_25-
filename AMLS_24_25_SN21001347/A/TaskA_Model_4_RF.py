from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from TaskA_utils import BreastMNISTDataset
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

# Step 1: Load and preprocess data
train_data = BreastMNISTDataset('train')
val_data = BreastMNISTDataset('val')
test_data = BreastMNISTDataset('test')

x_train = train_data.images.reshape(len(train_data.images), -1)/255.0
y_train = train_data.labels.ravel()
x_test = test_data.images.reshape(len(test_data.images), -1)/255.0
y_test = test_data.labels.ravel()

scaler = RobustScaler() #Scale data using RobustScaler
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Step 2: Apply SMOTE for class imbalance
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

# Step 3: Train Random Forest Classifier
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
rf.fit(x_train_resampled, y_train_resampled.ravel())

# Evaluate the model
y_pred = rf.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))