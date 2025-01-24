# AMLS_assignment24_25-

## File structure
```
AMLS_24_25_SN21001347/
├── A/
│   ├── Results/
│   ├── Saved Models/
│   ├── TaskA_CNN.py
│   ├── TaskA_FE.py
│   ├── TaskA_KNN.py
│   ├── TaskA_RF.py
│   ├── TaskA_SVM.py
│   └── TaskA_utils.py
├── B/
│   ├── Results/
│   ├── Saved Models/
│   ├── TaskB_CNN.py
│   ├── TaskB_FE.py
│   ├── TaskB_KNN.py
│   ├── TaskB_RF.py
│   ├── TaskB_SVM.py
│   └── TaskB_utils.py
├── Datasets/
├── README.md
└── main.py
```
## Project Description
The project consists of Tasks A and B. Some core files/folders include
- Task A
  - TaskA_CNN.py: Contains code to train and test the CNN model
  - TaskA_KNN.py: Contains code to train and test the KNN model
  - TaskA_RF.py: Contains code to train and test the RF model
  - TaskA_SWM.py: Contains code to train and test the SVM model
  - Saved_Models/: Contains saved CNN models
- Task B
  - TaskB_CNN.py: Contains code to train and test the CNN model
  - TaskB_KNN.py: Contains code to train and test the KNN model
  - TaskB_RF.py: Contains code to train and test the RF model
  - TaskB_SWM.py: Contains code to train and test the SVM model
  - Saved_Models/: Contains saved CNN models
- main.py: Main file to run all of the models used

## Packages used
numpy matplotlib torch torchvision torchaudio scikit-learn medmnist scipy pandas imbalanced-learn

## Instructions
- Download breastmnist.npz and bloodmnist.npz and place them into the Datasets/ folder
- Install the packages above and run the code through main.py
- Some lines of code used during testing were commented out, so feel free to change
