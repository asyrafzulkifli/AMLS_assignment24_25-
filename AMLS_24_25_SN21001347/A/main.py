import importlib

# Dispatcher dictionary
models = {
    "CNN": "A.TaskA_CNN.main",
    "KNN": "A.TaskA_KNN.main",
    "RF": "A.TaskA_RF.main",
    "SVM": "A.TaskA_SVM.main",
}

def run_model(model_name):
    if model_name in models:
        module_name, func = models[model_name].rsplit('.', 1)
        module = importlib.import_module(module_name)
        getattr(module, func)()
    else:
        print(f"Model {model_name} not found!")

def main():
    print("Available models:", list(models.keys()))
    model_name = input("Enter the model to run: ").strip()
    run_model(model_name)

if __name__ == "__main__":
    main()