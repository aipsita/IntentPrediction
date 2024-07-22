import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import sys
import os
from multiprocessing import Process, Manager
import time

# Function to preprocess new data
def preprocess_new_data(df, window_size, step_size, binning_factor):
    """
    Preprocess new data using sliding window and binning.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame with new data.
    window_size (int): The size of each sliding window.
    step_size (int): The step size for sliding the window.
    binning_factor (int): The factor by which to bin (average) the data in each window.
    
    Returns:
    pd.DataFrame: A DataFrame with the processed data.
    """
    # List to store the processed windows
    processed_data = []
    
    # Iterate over the data with the sliding window
    for start in range(0, len(df) - window_size + 1, step_size):
        # Extract the window
        window = df.iloc[start:start + window_size]
        
        # Separate features and output (assuming the last column is the output)
        #features = window
        #print(np.shape(window))
        
        # Apply binning by averaging for features
        # binned_features = window.groupby(window.index /% binning_factor).mean()
        binned_features = window.groupby(np.arange(len(window)) // 2).mean()
        #print(np.shape(binned_features))
        
        # Add the binned features to the processed data
        processed_data.append(binned_features)
    
    # Create a DataFrame from the processed data
    processed_df = pd.concat(processed_data, ignore_index=True)
    
    return processed_df

# Get the directory of the current script (test1.py)
current_directory = os.path.dirname(os.path.abspath(__file__))

# Add the directory containing test2.py to the Python path
sys.path.append(current_directory)

try:
    from Features1 import compute_features  # Import the function
    #print("Successfully imported process_data from test2.py")
except ImportError as e:
    print(f"Error importing process_data: {e}")
    
def main():
    
    data=[]
    features = pd.DataFrame()
    
    # Read data from a CSV file
    data_file = 'C:/Users/anany/Downloads/Feature/Data/Features/Study/data.csv'  # Update with your actual data file path
    data = pd.read_csv(data_file)
    
    start_time = time.time()
    # Create a new process to run the function from Features.py
    features = compute_features(data)
    new_data = features.drop(['event', 'Timestamp', 'time_difference'], axis=1)

    # Paths to the saved model and scaler
    model_file = 'C:/Users/anany/Downloads/Feature/Data/Features/Study/logistic_regression_model.pkl'
    scaler_file = 'C:/Users/anany/Downloads/Feature/Data/Features/Study/scaler.pkl'

    # Load the saved model and scaler
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    # Preprocess the new data
    window_size = 10
    step_size = 1
    binning_factor = 5
    processed_new_data = preprocess_new_data(new_data, window_size, step_size, binning_factor)
    #print(np.shape(processed_new_data))
    
    features_to_exclude = ['fixation_event', 'saccade_event', 'angular_displacement_saccade_point']
    
    # Drop the unwanted features
    processed_new_data = processed_new_data.drop(columns=features_to_exclude)
    
    # Define the paths
    folder_path = 'C:/Users/anany/Downloads/Feature/Data/Features/Study'
    output_file = os.path.join(folder_path, 'testing_processed_data.csv')
    processed_new_data.to_csv(output_file, index=False)
    #print(f'Saved processed data to {output_file}')
    
    # Scale the new data
    processed_new_data_scaled = scaler.transform(processed_new_data)
    
    # Make predictions on the new data
    predictions = model.predict_proba(processed_new_data_scaled)[:, 1]
    end_time = time.time()
    
    print(predictions)
    print(end_time-start_time)
    
def main_model_testing(data):
    
    #data=[]
    features = pd.DataFrame()
    
    # Read data from a CSV file
    #data_file = 'C:/Users/anany/Downloads/Feature/Data/Features/Study/data.csv'  # Update with your actual data file path
    #data = pd.read_csv(data_file)
    
    start_time = time.time()
    # Create a new process to run the function from Features.py
    #print(len(data))
    features = compute_features(data)
    new_data = features.drop(['event', 'Timestamp', 'time_difference'], axis=1)

    # Paths to the saved model and scaler
    model_file = 'C:/Users/anany/Downloads/Feature/Data/Features/Study/Features/logistic_regression_model.pkl'
    scaler_file = 'C:/Users/anany/Downloads/Feature/Data/Features/Study/Features/scaler.pkl'

    # Load the saved model and scaler
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    # Preprocess the new data
    window_size = 10
    step_size = 1
    binning_factor = 5
    processed_new_data = preprocess_new_data(new_data, window_size, step_size, binning_factor)
    #print(np.shape(processed_new_data))
    
    features_to_exclude = ['fixation_event', 'saccade_event']
    
    # Drop the unwanted features
    processed_new_data = processed_new_data.drop(columns=features_to_exclude)
    
    # Define the paths
    #folder_path = 'C:/Users/anany/Downloads/Feature/Data/Features/Study'
    #output_file = os.path.join(folder_path, 'testing_processed_data.csv')
    #processed_new_data.to_csv(output_file, index=False)
    #print(f'Saved processed data to {output_file}')
    
    # Scale the new data
    processed_new_data_scaled = scaler.transform(processed_new_data)
    
    # Make predictions on the new data
    predictions = model.predict_proba(processed_new_data_scaled)[:, 1]
    end_time = time.time()
    
    print(predictions)
    print(end_time-start_time)
    
    return predictions[-1]

if __name__ == "__main__":
    print("Running main function")
    main()
