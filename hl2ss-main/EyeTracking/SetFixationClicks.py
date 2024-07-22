import pandas as pd
import pandas as pd
import numpy as np
import scipy.spatial.distance

def vector_dispersion(vectors):
    distances = scipy.spatial.distance.pdist(vectors, metric='cosine')
    distances.sort()
    cut_off = max(distances.shape[0] // 5, 4)
    radians = np.arccos(1.0 - distances[-cut_off:].mean())
    degrees = np.degrees(radians)
    return degrees

# Load the CSV file
folder_path = 'C:/Users/anany/Downloads/Feature/Data-1/User6/Study2/'
csv_file = 'C:/Users/anany/Downloads/Feature/Data-1/User6/Study2/20240708_151337797-data.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file)

# Assuming 'common_indices' is the list of indices for invalid key presses
common_indices = [250, 1512]  # Replace with actual invalid indices list

# Initialize column E with zeros
data['is_fixation'] = 0

# Create a helper column 'd' to mark the rows where b[index] = 1
data['valid_clicks'] = data['KeyPress'].apply(lambda x: 1 if x == '1' else 0)

# Set the invalid key presses to 0 in 'valid Key presses'
data.loc[common_indices, 'valid_clicks'] = 0

# Iterate through the DataFrame to populate column E
for index in range(len(data)):
    if data.loc[index, 'valid_clicks'] == 1:
        start_index = max(0, index - 6)
        data.loc[start_index:index, 'is_fixation'] = 1

# Drop the helper column 'd'
data.drop(columns=['valid_clicks'], inplace=True)

# Save the updated DataFrame back to a CSV file
#output_file = folder_path + '20240626_082556300-data_output_file.csv'  # Replace with your desired output file path
#data.to_csv(output_file, index=False)

#print("Column E has been populated based on the given condition.")

# Filter fixation samples
fixation_data = data[data['is_fixation'] == 1].reset_index(drop=True)
print(np.shape(fixation_data))

# Initialize a list to store dispersion values
dispersion_values = []

# Iterate through the fixation data in chunks of 20 samples
for i in range(0, len(fixation_data), 7):
    if i + 7 <= len(fixation_data):
        # Get the current chunk of 20 samples
        chunk = fixation_data.iloc[i:i+7]
        
        # Compute vectors from gaze origin to target coordinates
        #fixation_vectors = chunk[['Gaze.x', 'Gaze.y', 'Gaze.z']].values - chunk[['TargetPosition.x', 'TargetPosition.y', 'TargetPosition.z']].values
        fixation_vectors = chunk[['Gazedir.x', 'Gazedir.y', 'Gazedir.z']].values
        
        # Calculate vector dispersion for the current chunk
        vector_disp = vector_dispersion(fixation_vectors)
        
        # Store the dispersion value in the list
        dispersion_values.append(vector_disp)

# Print the dispersion values
print("Dispersion values for each fixation:")
for i, disp in enumerate(dispersion_values):
    print(f"Fixation {i+1}: {disp}")
    
# Calculate the average of the dispersion values
average_dispersion = np.mean(dispersion_values)

# Print the average
print(f"Average Dispersion: {average_dispersion}")