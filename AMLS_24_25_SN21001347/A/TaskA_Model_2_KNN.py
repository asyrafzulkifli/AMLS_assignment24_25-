from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
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

# Step 3: Perform GridSearch to optimize 'n_neighbors'
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
grid = GridSearchCV(KNeighborsClassifier(weights='distance', metric='euclidean'),
                    param_grid, cv=5, scoring='f1_weighted')
grid.fit(x_train_resampled, y_train_resampled)

# Step 4: Train KNN with the best parameters
best_params = grid.best_params_
print("Best Parameters:", best_params)

best_knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], weights='distance', metric='euclidean')
best_knn.fit(x_train_resampled, y_train_resampled)

# Step 5: Evaluate the model
y_pred = best_knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))