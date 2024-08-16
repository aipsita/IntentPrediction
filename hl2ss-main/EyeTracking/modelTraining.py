import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
import joblib  # For saving the model and scaler

# Define the paths
folder_path = 'C:/Users/anany/Downloads/Feature/Data/Features/Study/Features'
output_file = os.path.join(folder_path, 'processed_data.csv')
model_file = os.path.join(folder_path, 'logistic_regression_model.pkl')
scaler_file = os.path.join(folder_path, 'scaler.pkl')

# Check if processed data already exists
if os.path.exists(output_file):
    processed_data = pd.read_csv(output_file)
    print(f'Loaded processed data from {output_file}')
else:
    file_names = [f for f in os.listdir(folder_path) if f.endswith('_features.csv')]
    processed_data_frames = []

    # Step 2: Preprocess Data (implement sliding window and binning)
    def apply_sliding_window(df, window_size, step_size, binning_factor, output_column='transition_int'):
        """
        Apply sliding window to a DataFrame with given window size, step size, and binning factor.
        
        Parameters:
        df (pd.DataFrame): The input DataFrame with time-series data.
        window_size (int): The size of each sliding window.
        step_size (int): The step size for sliding the window.
        binning_factor (int): The factor by which to bin (average) the data in each window.
        
        Returns:
        pd.DataFrame: A new DataFrame with the processed data.
        """
        # List to store the processed windows
        processed_data = []
        
        print(f'Original data shape: {df.shape}')
        
        # Iterate over the data with the sliding window
        for start in range(0, len(df) - window_size + 1, step_size):
            # Extract the window
            window = df.iloc[start:start + window_size]
            #print(f'Window shape: {window.shape}')
            
            # Separate features and output
            features = window.drop(columns=[output_column])
            output = window[output_column]
            
            # Apply binning by averaging for features
            binned_features = features.groupby(np.arange(len(window)) // 2).mean()
            
            # Use the last element for the output
            binned_output = output.iloc[-1]
            
            # Combine binned features and output
            binned_window = binned_features
            binned_window[output_column] = binned_output
            
            # Add the binned window to the processed data
            processed_data.append(binned_window)
        
        # Create a DataFrame from the processed data
        processed_df = pd.concat(processed_data, ignore_index=True)
        print(f'Processed data shape: {processed_df.shape}')
        
        return processed_df

    window_size = 10
    step_size = 1
    binning_factor = 5

    for file in file_names:
        df = pd.read_csv(os.path.join(folder_path, file))
        print(f'Processing file: {file}')
        df = df.drop(['event', 'Timestamp', 'time_difference'], axis=1)
        processed_df = apply_sliding_window(df, window_size, step_size, binning_factor)
        processed_data_frames.append(processed_df)

    processed_data = pd.concat(processed_data_frames, ignore_index=True)
    processed_data.to_csv(output_file, index=False)
    print(f'Saved processed data to {output_file}')

print(f'Final processed data shape: {processed_data.shape}')

# Define the features to be used during training
input_features = ['is_fixation_int', 'gaze_velocity', 'fixation_velocity_mean',
                  'saccade_gaze_acceleration_x_skew', 'saccade_y_std', 'saccade_gaze_velocity_kurtosis',
                  'saccade_gaze_velocity_skew', 'saccade_gaze_velocity_x_skew', 'saccade_duration',
                  'k_coefficient', 'saccade_gaze_velocity_std_dev', 'angular_displacement_saccade_point']  # Replace with actual feature names

# Step 3: Train-test split
#X = processed_data[input_features]
#X = processed_data.drop('transition_int', axis=1)
# Exclude specific features
features_to_exclude = ['fixation_event', 'saccade_event']

# Drop the unwanted features
X = processed_data.drop(columns=features_to_exclude + ['transition_int'])
y = processed_data['transition_int']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, scaler_file)
print(f'Saved scaler to {scaler_file}')

# Check class distribution in train and test sets
print("Class distribution in y_train:", y_train.value_counts())
print("Class distribution in y_test:", y_test.value_counts())

if len(y_train.unique()) > 1:
    # Train logistic regression model with class weights
    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Save the model
    joblib.dump(model, model_file)
    print(f'Saved model to {model_file}')
    
    y_pred = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_class = model.predict(X_test_scaled)
    
    # ROC AUC and PR AUC
    roc_auc = roc_auc_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    
    print(f'ROC AUC: {roc_auc}')
    print(f'PR AUC: {pr_auc}')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot()
    plt.show()
    
    # Classification Report
    report = classification_report(y_test, y_pred_class, target_names=['Class 0', 'Class 1'])
    print(report)
    
    # Sensitivity (Recall) and Specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print(f'Sensitivity (Recall): {sensitivity}')
    print(f'Specificity: {specificity}')
    
    # Plot ROC and PR curves
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(recall, precision, label=f'PR AUC: {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Precision-Recall Curve')
    
    plt.subplot(2, 1, 2)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label=f'ROC AUC: {roc_auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('ROC Curve')
    
    plt.tight_layout()
    plt.show()
    
else:
    print("Not enough classes to train the model. Please check the data.")
