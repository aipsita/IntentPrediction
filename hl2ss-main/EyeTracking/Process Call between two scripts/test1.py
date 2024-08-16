import sys
import os
from multiprocessing import Process, Manager

# Get the directory of the current script (test1.py)
current_directory = os.path.dirname(os.path.abspath(__file__))

# Print the current directory for debugging purposes
print(f"Current directory: {current_directory}")

# Add the directory containing test2.py to the Python path
sys.path.append(current_directory)

# Print the sys.path for debugging purposes
print(f"sys.path: {sys.path}")

try:
    from test2duplicate import process_data  # Import the function
    print("Successfully imported process_data from test2.py")
except ImportError as e:
    print(f"Error importing process_data: {e}")

def main():
    # Create a Manager object to handle shared data
    manager = Manager()
    shared_data = manager.dict()

    # Create a new process to run the function from test2.py
    p = Process(target=process_data, args=(shared_data,))
    print("Starting process")
    p.start()

    # Wait for the process to finish
    p.join()
    print("Process finished")

    # Retrieve the result from the shared dictionary
    result = shared_data['result']
    print(f"Result from test2.py: {result}")

if __name__ == "__main__":
    print("Running main function")
    main()
