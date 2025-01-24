import importlib

# Dispatcher dictionary: Maps task names (keys) to the corresponding module and function (values)
tasks = {
    "A": "A.main.main", # Task "A" maps to the `main` function in `A/main.py`
    "B": "B.main.main", # Task "B" maps to the `main` function in `B/main.py`
}

# Function to dynamically run tasks based on user input
def run_task(task_name):
    if task_name in tasks: # Check if the task exists in the dispatcher dictionary
        # Split the module path and function name
        module_name, task = tasks[task_name].rsplit('.', 1)
        # Exception handling for import errors
        try:
            # Dynamically import the specified module and call the specified function
            module = importlib.import_module(module_name)
            getattr(module, task)()
        except ImportError:
            print(f"Error: Could not import module {module_name}.")
            main() # Returns to main menu
        except AttributeError:
            print(f"Error: Function {task} not found in module {module_name}.")
            main() # Returns to main menu
    elif task_name == "0":
        print("Exiting...")
    else:
        print(f"Function {task_name} not found!")
        main() # Returns to main menu

# Main menu
def main():
    print("Available tasks:", list(tasks.keys()),"\nEnter 0 to exit.")
    task_name = input("Enter the task to run: ").strip() # User input for task selection
    run_task(task_name)

if __name__ == "__main__":
    main()
