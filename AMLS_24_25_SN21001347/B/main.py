import importlib

# Dispatcher dictionary 
models = {
    "CNN": "B.TaskB_CNN.main",
    "KNN": "B.TaskB_KNN.main",
    "RF": "B.TaskB_RF.main",
    "SVM": "B.TaskB_SVM.main",
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
