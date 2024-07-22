import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Optionally, convert the list of fixations to a DataFrame for easier manipulation and saving
import pandas as pd
import scipy.spatial.distance
import numpy as np
from scipy.stats import zscore, skew, kurtosis
import warnings 

# spatial threshold or the dispersion itself
max_dispersion = np.deg2rad(0.55)
# temporal threshold or duration
min_duration = 1000000 # 100 ms
max_duration = 20000000  # 400 ms

# Constants
R = 10  # Radius of the sphere, can adjust as needed
SAMPLE_RATE = 90  # Adjust the sample rate as needed
TIME_STEP = 1 / SAMPLE_RATE
NS_PER_UNIT = 100  # Each unit in timestamp represents 100 nanoseconds
SECONDS_PER_UNIT = NS_PER_UNIT * 1e-9  # Convert units to seconds
RADIANS_TO_DEGREES = 180 / np.pi
WINDOW_LENGTH = 10  # Adjust based on your analysis needs
SACCADE_VELOCITY_THRESHOLD = 10  # degrees per second, according to your theoretical model
SECONDS_PER_UNIT = 100 * 1e-9  # Convert units to seconds, assuming each unit is 100 nanoseconds
# Define minimum and maximum saccade durations in seconds
MIN_SACCADE_DURATION = 17 / 1000  # minimum saccade duration in seconds
MAX_SACCADE_DURATION = 200 / 1000  # maximum saccade duration in seconds
# Constants
ONE_SECOND = 10000000  # One second in 100 ns units

def angular_distance(v1, v2):
    """Calculate the angular distance between two vectors."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    cosine_similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cosine_distance = 1 - cosine_similarity
    return np.arccos(np.clip(cosine_distance, -1.0, 1.0)) * (180 / np.pi)

def modified_gaze_dispersion(eye_data):
    if len(eye_data) < 2:
        return float("inf")
    centroid = get_centroid(eye_data)
    max_distance = max(angular_distance(centroid, (p['gazeDirection_x'], p['gazeDirection_y'], p['gazeDirection_z'])) for p in eye_data)
    return max_distance

def vector_dispersion(vectors):
    distances = scipy.spatial.distance.pdist(vectors, metric='cosine')
    distances.sort()
    cut_off = np.max([distances.shape[0] // 5, 4])
    return np.arccos(1. - distances[-cut_off:].mean())

def gaze_dispersion(eye_data):
    base_data = eye_data

    vectors = []
    for p in base_data:
        vectors.append((p['gazeDirection_x'], p['gazeDirection_y'], p['gazeDirection_z']))
    vectors = np.array(vectors, dtype=np.float32)

    if len(vectors) < 2:
        return float("inf")
    else:
        return vector_dispersion(vectors)  


def get_centroid(eye_data):
    '''Calculates the centroid for each point in a df of points.
    Input: Df of points.
    Output: Vector containg the centroid of all points.'''
    x = [p['gazeDirection_x'] for p in eye_data]
    y = [p['gazeDirection_y'] for p in eye_data]
    z = [p['gazeDirection_z'] for p in eye_data]
    return (sum(x) / len(eye_data), sum(y) / len(eye_data), sum(z) / len(eye_data))

from collections import deque

def detect_fixations(gaze_data):
    # Convert Pandas data frame to list of Python dictionaries
    gaze_data = gaze_data.T.to_dict().values()

    candidate = deque()
    future_data = deque(gaze_data)
    while future_data:
        # check if candidate contains enough data
        if len(candidate) < 2 or ((candidate[-1]['eyeDataTimestamp'] - candidate[0]['eyeDataTimestamp']) < min_duration):
            datum = future_data.popleft()
            if datum['gaze_velocity'] < SACCADE_VELOCITY_THRESHOLD:
                candidate.append(datum)
            else:
                candidate.clear()  # Clear the candidate if velocity is too high
            continue

        # Minimal duration reached, check for fixation
        dispersion = gaze_dispersion(candidate)
        if dispersion > max_dispersion:
            # not a fixation, move forward
            candidate.popleft()
            continue

        # Minimal fixation found. Try to extend!
        while future_data:
            datum = future_data[0]
            candidate.append(datum)
            if datum['gaze_velocity'] > SACCADE_VELOCITY_THRESHOLD:
                # High velocity point
                candidate.pop()
                break  # End current fixation checking and start fresh

            dispersion = gaze_dispersion(candidate)
            if dispersion > max_dispersion:
                # end of fixation found
                candidate.pop()
                break
            else:
                # still a fixation, continue extending
                future_data.popleft()
        centroid = get_centroid(candidate)
        yield {"fixation_start": candidate[0]['eyeDataTimestamp'], "fixation_end": candidate[-1]['eyeDataTimestamp'],
               "fixation_duration": (candidate[-1]['eyeDataTimestamp'] - candidate[0]['eyeDataTimestamp'])*SECONDS_PER_UNIT,
              "fixation_centroid": centroid, "fixation_dispersion": dispersion}
        candidate.clear()


import pandas as pd

def only_valid_data(data):
    '''Returns only valid gaze points. Those have values in gazeDirection_x etc.'''
    return data[(data.gazeHasValue == True) & (data.isCalibrationValid == True)]

# Function to clean and convert space-separated strings to numpy arrays
def parse_array_string(array_string):
    try:
        cleaned_string = ' '.join(array_string.strip().split())
        cleaned_string = cleaned_string.replace(' ', ',')
        cleaned_string = array_string.replace('[ ', '[').replace(' ]', ']').replace(' ', ',').replace(',,', ',')
        cleaned_string = cleaned_string.replace('[,', '[').replace(',]', ']').replace(',,', ',')
        cleaned_string = cleaned_string.strip('[],')
        cleaned_string = cleaned_string.replace(',,', ',')
        cleaned_string = '[' + cleaned_string + ']'
        return np.array(eval(cleaned_string))
    except Exception as e:
        print(f"Error parsing string: {array_string}")
        raise e
        
# Function to parse the 4x4 pose matrix
def parse_pose_matrix(pose_string):
    try:
        # Remove outer brackets and extra spaces
        pose_string = pose_string.replace('[', '').replace(']', '').strip()
        # Split into individual elements
        elements = [float(x) for x in pose_string.split()]
        # Ensure it's a 4x4 matrix
        if len(elements) != 16:
            raise ValueError(f"Expected 16 elements for a 4x4 matrix, got {len(elements)} elements")
        return np.array(elements).reshape((4, 4))
    except Exception as e:
        print(f"Error parsing pose matrix: {pose_string}")
        raise e

# Transform gaze origin and direction using head orientation (pose matrix)
def transform_gaze(pose_matrix, gaze_origin, gaze_direction):
    rotation_matrix = pose_matrix[:3, :3]  # Extract rotation part    
    # Convert gaze_origin to homogeneous coordinates
    homogeneous_gaze_origin = np.append(gaze_origin, 1.0)  
    # Apply full transformation to the gaze origin
    transformed_gaze_origin = homogeneous_gaze_origin @ pose_matrix   
    # Apply rotation only to the gaze direction (no translation for directions)
    transformed_gaze_direction = gaze_direction @ rotation_matrix  
    return transformed_gaze_origin, transformed_gaze_direction

def compute_intersection(origin, direction):
    # Ensure both vectors are 3D if necessary
    origin = np.array(origin[:3])  # Use only the first three components if it's 4D
    direction = np.array(direction)  # Ensure direction is also 3D if not already

    # Calculate the parameter s for the intersection point on the sphere
    a = 1
    b = 2 * np.dot(origin, direction)
    c = np.dot(origin, origin) - R**2
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # No real solution, can occur if the origin is inside the sphere or due to rounding errors
        return None  # Handle cases where there's no intersection
    
    s = (-b + np.sqrt(discriminant)) / (2 * a)  # Choosing the positive root
    return origin + s * direction

# Function to compute saccade amplitude
def compute_saccade_amplitude(data):
    # Group by saccade event
    grouped = data.groupby('saccade_event')
    amplitudes = {}
    
    for name, group in grouped:
        if name == 0:
            # Skip processing for saccade_event 0
            continue
        if len(group) > 1:
            # Calculate the Euclidean distance between the first and last point of each saccade
            start_x, start_y = group.iloc[0]['gazeOrigin_x'], group.iloc[0]['gazeOrigin_y']
            end_x, end_y = group.iloc[-1]['gazeOrigin_x'], group.iloc[-1]['gazeOrigin_y']
            amplitude = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            amplitudes[name] = amplitude
        else:
            amplitudes[name] = np.nan  # No amplitude for single-point "saccades"

    # Map amplitudes back to the DataFrame
    data['saccade_amplitude'] = data['saccade_event'].map(amplitudes).fillna(0)
    return data

# Function to calculate path length for fixations
def calculate_path_length(x, y):
    return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))

def angular_displacement(v1, v2):
    """Calculate angular displacement between two vectors, handling different dimensions."""
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return np.nan  # Return NaN if either vector is zero-length to avoid division by zero
    
    if v1.shape != v2.shape:
        print(f"Dimension mismatch: v1={v1.shape}, v2={v2.shape}")
        return np.nan  # Return NaN if dimensions mismatch

    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(np.clip(dot_product, -1.0, 1.0)) * (180 / np.pi)  # Convert radians to degrees and clip for safety

def compute_M3S2K(series):
    results = {
        'mean': series.mean(),
        'median': series.median(),
        'max': series.max(),
        'std_dev': series.std(),
        'skew': series.skew(),
        'kurtosis': series.kurtosis()
    }
    return pd.Series(results)

# Function to compute z-scores within a custom window
def custom_window_zscore(index, data, column):
    # Find the start time for the 5s window
    current_time = data.at[index, 'cumulative_time']
    window_start_time = current_time - 5

    # Filter data for the current window
    window_data = data[(data['cumulative_time'] <= current_time) & (data['cumulative_time'] > window_start_time)]
    
    # Compute the z-score for the data in the window
    if not window_data.empty:
        mean_val = window_data[column].mean()
        std_val = window_data[column].std()
        current_val = data.at[index, column]
        if std_val > 0:
            return (current_val - mean_val) / std_val
    return 0  # Return 0 or some other neutral value if the window is too small for statistics

def compute_feature(data):
    # Assuming data contains 'fixation_duration' and 'saccade_amplitude' along with 'gaze_x' and 'gaze_y' for positions

    # Calculating mean and std deviation for fixations and saccades
    data['fixation_x_mean'] = data.groupby('fixation_event')['gazeOrigin_x'].transform('mean')
    data['fixation_y_mean'] = data.groupby('fixation_event')['gazeOrigin_y'].transform('mean')
    data['fixation_x_std'] = data.groupby('fixation_event')['gazeOrigin_x'].transform('std')
    data['fixation_y_std'] = data.groupby('fixation_event')['gazeOrigin_y'].transform('std')

    data['saccade_x_mean'] = data.groupby('saccade_event')['gazeOrigin_x'].transform('mean')
    data['saccade_y_mean'] = data.groupby('saccade_event')['gazeOrigin_y'].transform('mean')
    data['saccade_x_std'] = data.groupby('saccade_event')['gazeOrigin_x'].transform('std')
    data['saccade_y_std'] = data.groupby('saccade_event')['gazeOrigin_y'].transform('std')

    # Skewness and Kurtosis
    data['fixation_x_skew'] = data.groupby('fixation_event')['gazeOrigin_x'].transform(skew)
    data['fixation_y_skew'] = data.groupby('fixation_event')['gazeOrigin_y'].transform(skew)
    data['fixation_x_kurt'] = data.groupby('fixation_event')['gazeOrigin_x'].transform(kurtosis)
    data['fixation_y_kurt'] = data.groupby('fixation_event')['gazeOrigin_y'].transform(kurtosis)
    
    # Assuming 'data' is your DataFrame
    data = compute_saccade_amplitude(data)

    # K Coefficient calculation
    #data['z_fixation_duration'] = zscore(data['fixation_duration'])
    #data['z_saccade_amplitude'] = zscore(data['saccade_amplitude'])
    #data['k_coefficient'] = data['z_fixation_duration'] - data['z_saccade_amplitude']    
    # Calculate cumulative time
    data['cumulative_time'] = data['time_difference'].cumsum()
    # Calculate z-scores for each sample using the custom window
    data['z_fixation_duration'] = data.apply(lambda x: custom_window_zscore(x.name, data, 'fixation_duration'), axis=1)
    data['z_saccade_amplitude'] = data.apply(lambda x: custom_window_zscore(x.name, data, 'saccade_amplitude'), axis=1)
    # Compute the k-coefficient
    data['k_coefficient'] = data['z_fixation_duration'] - data['z_saccade_amplitude']
    
    # Path length for fixations
    data['fixation_path_length'] = data.groupby('fixation_event').apply(
        lambda df: calculate_path_length(df['gazeOrigin_x'], df['gazeOrigin_y'])
    ).reset_index(level=0, drop=True)
    
    # Path length for saccades
    data['saccade_path_length'] = data.groupby('saccade_event').apply(
        lambda df: calculate_path_length(df['gazeOrigin_x'], df['gazeOrigin_y'])
    ).reset_index(level=0, drop=True)
    
    # Calculate the last fixation centroid at the end of each event
    data['last_fixation_centroid'] = data.groupby('fixation_event')['fixation_centroid'].transform('last')
    # Define the previous valid fixation event
    data['previous_valid_fixation_event'] = data['fixation_event'].replace(0, np.nan).ffill().shift()
    # Pre-calculate the mapping dictionary once to avoid resetting the index in the map function
    centroid_mapping = data[data['fixation_event'] != 0].drop_duplicates('fixation_event', keep='last').set_index('fixation_event')['last_fixation_centroid'].to_dict()
    # Create a mask to determine when a new fixation event starts
    new_fixation_event_mask = data['fixation_event'] != data['fixation_event'].shift(1)
    # Map the last fixation centroid from the previous event using the mask and the pre-calculated dictionary
    data['previous_fixation_centroid'] = np.where(new_fixation_event_mask, data['previous_valid_fixation_event'].map(centroid_mapping), np.nan)
    # Propagate this value for all rows within the same fixation event, ignoring zeros
    data['previous_fixation_centroid'] = data['previous_fixation_centroid'].ffill()
    data['previous_fixation_centroid'] = data['previous_fixation_centroid'].where(data['is_fixation'], np.nan)
    data['fixation_angular_displacement_centroid'] = data.apply(
        lambda row: angular_displacement(np.array(row['fixation_centroid']), np.array(row['previous_fixation_centroid']))
        if isinstance(row['fixation_centroid'], tuple) and isinstance(row['previous_fixation_centroid'], tuple) else np.nan, 
        axis=1
    )
    
    # Get the last sample vector of the previous fixation
    data['last_sample_x'] = data.groupby('fixation_event')['gazeOrigin_x'].transform('last')
    data['last_sample_y'] = data.groupby('fixation_event')['gazeOrigin_y'].transform('last')
    data['last_sample_z'] = data.groupby('fixation_event')['gazeOrigin_z'].transform('last')
    # Create vector for the last sample of each fixation directly
    data['last_sample_vector'] = list(zip(data['last_sample_x'], data['last_sample_y'], data['last_sample_z']))
    # Convert tuples in 'last_sample_vector' to np.array
    data['last_sample_vector'] = data['last_sample_vector'].apply(np.array)    
    # Pre-calculate the mapping dictionary once to avoid resetting the index in the map function
    sample_mapping = data[data['fixation_event'] != 0].drop_duplicates('fixation_event', keep='last').set_index('fixation_event')['last_sample_vector'].to_dict()
    # Map the last fixation centroid from the previous event using the mask and the pre-calculated dictionary
    data['previous_fixation_sample'] = np.where(new_fixation_event_mask, data['previous_valid_fixation_event'].map(sample_mapping), np.nan)
    # Propagate this value for all rows within the same fixation event, ignoring zeros
    data['previous_fixation_sample'] = data['previous_fixation_sample'].ffill()
    data['previous_fixation_sample'] = data['previous_fixation_sample'].where(data['is_fixation'], np.nan)
    # Calculate angular displacement between fixation centroid and last sample vector of previous fixation
    data['angular_displacement_last_sample'] = data.apply(
        lambda row: angular_displacement(
            np.array(row['fixation_centroid']) if not isinstance(row['fixation_centroid'], np.ndarray) else row['fixation_centroid'],
            np.array(row['previous_fixation_sample']) if not isinstance(row['previous_fixation_sample'], np.ndarray) else row['previous_fixation_sample']
        ) if (isinstance(row['previous_fixation_sample'], (np.ndarray, tuple)) and 
            isinstance(row['fixation_centroid'], (np.ndarray, tuple)) and
            not np.isnan(row['previous_fixation_sample']).any() and 
            not np.isnan(row['fixation_centroid']).any()) else np.nan, 
        axis=1
    )

    # Drop the helper columns if necessary
    data.drop(columns=['last_sample_x', 'last_sample_y', 'last_sample_z', 'last_sample_vector', 'previous_valid_fixation_event', 'previous_fixation_centroid', 'previous_fixation_sample'], inplace=True)

    # Saccadic ratio: Peak velocity / Duration
    data['saccade_peak_velocity'] = data.groupby('saccade_event')['gaze_velocity'].transform('max')
    data['saccadic_ratio'] = data['saccade_peak_velocity'] / data['saccade_duration']
    
    # Get the last sample vector of the previous saccade
    data['last_saccade_sample_x'] = data.groupby('saccade_event')['gazeOrigin_x'].transform('last')
    data['last_saccade_sample_y'] = data.groupby('saccade_event')['gazeOrigin_y'].transform('last')
    data['last_saccade_sample_z'] = data.groupby('saccade_event')['gazeOrigin_z'].transform('last')
    # Create vector for the last sample of each saccade directly
    data['last_saccade_sample_vector'] = list(zip(data['last_saccade_sample_x'], data['last_saccade_sample_y'], data['last_saccade_sample_z']))
    # Convert tuples in 'last_saccade_sample_vector' to np.array
    data['last_saccade_sample_vector'] = data['last_saccade_sample_vector'].apply(np.array)  
    # Define the previous valid saccade event
    data['previous_valid_saccade_event'] = data['saccade_event'].replace(0, np.nan).ffill().shift()
    # Pre-calculate the mapping dictionary once to avoid resetting the index in the map function
    saccade_sample_mapping = data[data['saccade_event'] != 0].drop_duplicates('saccade_event', keep='last').set_index('saccade_event')['last_saccade_sample_vector'].to_dict()
    # Create a mask to determine when a new fixation event starts
    new_saccade_event_mask = data['saccade_event'] != data['saccade_event'].shift(1)
    # Map the last saccade centroid from the previous event using the mask and the pre-calculated dictionary
    data['previous_saccade_sample'] = np.where(new_saccade_event_mask, data['previous_valid_saccade_event'].map(saccade_sample_mapping), np.nan)
    # Propagate this value for all rows within the same saccade event, ignoring zeros
    data['previous_saccade_sample'] = data['previous_saccade_sample'].ffill()
    data['previous_saccade_sample'] = data['previous_saccade_sample'].where(data['is_saccade'], np.nan)
    data['last_saccade_sample_vector'] = data['last_saccade_sample_vector'].where(data['is_saccade'], np.nan)
    # Calculate angular displacement between fixation centroid and last sample vector of previous fixation
    data['angular_displacement_saccade_point'] = data.apply(
        lambda row: angular_displacement(
            np.array(row['previous_saccade_sample']) if not isinstance(row['previous_saccade_sample'], np.ndarray) else row['previous_saccade_sample'],
            np.array(row['last_saccade_sample_vector']) if not isinstance(row['last_saccade_sample_vector'], np.ndarray) else row['last_saccade_sample_vector']
        ) if (isinstance(row['previous_saccade_sample'], (np.ndarray, tuple)) and 
            isinstance(row['last_saccade_sample_vector'], (np.ndarray, tuple)) and
            not np.isnan(row['previous_saccade_sample']).any() and 
            not np.isnan(row['last_saccade_sample_vector']).any()) else np.nan, 
        axis=1
    )
    
    # Drop the helper columns if necessary
    data.drop(columns=['last_saccade_sample_x', 'last_saccade_sample_y', 'last_saccade_sample_z', 'last_saccade_sample_vector', 'previous_valid_saccade_event', 'previous_saccade_sample'], inplace=True)

    # Calculate velocities
    data['gaze_velocity_x'] = data['gazeOrigin_x'].diff() / data['time_difference']
    data['gaze_velocity_y'] = data['gazeOrigin_y'].diff() / data['time_difference']

    # Calculate acceleration
    data['gaze_acceleration_x'] = data['gaze_velocity_x'].diff() / data['time_difference']
    data['gaze_acceleration_y'] = data['gaze_velocity_y'].diff() / data['time_difference']
    
    # M3S2K calculations for saccades
    for feature in ['gaze_velocity', 'gaze_acceleration', 'gaze_velocity_x', 'gaze_velocity_y', 'gaze_acceleration_x', 'gaze_acceleration_y']:
        # Apply the M3S2K function to saccade data
        saccade_stats = data.groupby('saccade_event')[feature].apply(compute_M3S2K).unstack()
        for stat in ['mean', 'median', 'max', 'std_dev', 'skew', 'kurtosis']:
            data[f'saccade_{feature}_{stat}'] = data['saccade_event'].map(saccade_stats[stat])

    # M3S2K calculations for fixations: only applying to velocity for simplicity
    fixation_stats = data.groupby('fixation_event')['gaze_velocity'].apply(compute_M3S2K).unstack()
    for stat in ['mean', 'median', 'max', 'std_dev', 'skew', 'kurtosis']:
        data[f'fixation_velocity_{stat}'] = data['fixation_event'].map(fixation_stats[stat])

    # Ensure all new columns are filled properly
    data.fillna(0, inplace=True)

def compute_features(data):
    # Load the data
    #data = pd.read_csv('C:/Users/anany/Downloads/Feature/Data/study1.csv')
    #print("hello")
    
    # Settings the warnings to be ignored 
    warnings.filterwarnings('ignore') 

    features = pd.DataFrame()

    # Apply the function to the appropriate columns
    #print(data['Combined_Eye_Gaze_Origin'].head(5))
    data['gaze_origin'] = data['Combined_Eye_Gaze_Origin'] #.apply(parse_array_string)
    data['gaze_direction'] = data['Combined_Eye_Gaze_Direction'] #.apply(parse_array_string)
    data['pose_matrix'] = data['Pose'] #.apply(parse_pose_matrix)

    # Apply the transformation
    transformed_gaze = data.apply(lambda row: transform_gaze(row['pose_matrix'], row['gaze_origin'], row['gaze_direction']), axis=1)
    data['world_gaze_origin'] = transformed_gaze.apply(lambda x: x[0])
    data['world_gaze_direction'] = transformed_gaze.apply(lambda x: x[1])

    data['isCalibrationValid'] = data['Calibration_Valid']
    data['gazeHasValue'] = data['Combined_Eye_Gaze_Valid']
    data = only_valid_data(data)

    # Remove entries where time_diff indicates duplicates or significant data gaps
    data = data[(data['Timestamp'].diff().fillna(0) > 1000)]
    
    # Normalize gaze directions to remove magnitude effects
    #data['gaze_direction_normalized'] = data['world_gaze_direction'].apply(lambda x: x / np.linalg.norm(x))
    data['gaze_direction_normalized'] = data['world_gaze_direction']

    # Ensure there are no NaN values in gaze_direction_normalized before shifting
    data['next_gaze_direction'] = data['gaze_direction_normalized'].shift(-1)

    # Drop rows where next_gaze_direction is NaN
    data = data.dropna(subset=['next_gaze_direction'])

    # Now calculate theta safely
    data['theta'] = data.apply(lambda row: 2 * np.arctan2(
        np.linalg.norm(row['gaze_direction_normalized'] - row['next_gaze_direction']),
        np.linalg.norm(row['gaze_direction_normalized'] + row['next_gaze_direction'])), axis=1)
    #print(data['theta'].head(20))

    '''
    # Normalize gaze directions to remove magnitude effects
    data['gaze_direction_normalized'] = data['world_gaze_direction'].apply(lambda x: x / np.linalg.norm(x))
    
    data['sphere_intersection'] = data.apply(lambda row: compute_intersection(row['world_gaze_origin'], row['gaze_direction_normalized']), axis=1)
    
    # Drop rows where next_sphere_intersection is NaN (last row typically)
    data = data.dropna(subset=['sphere_intersection'])
    
    # Compute angles using dot product
    data['next_sphere_intersection'] = data['sphere_intersection'].shift(-1)
    
    # Drop rows where next_sphere_intersection is NaN (last row typically)
    data = data.dropna(subset=['next_sphere_intersection'])
    
    data['cos_theta'] = data.apply(lambda row: np.dot(row['sphere_intersection'], row['next_sphere_intersection']) / (np.linalg.norm(row['sphere_intersection']) * np.linalg.norm(row['next_sphere_intersection'])), axis=1)
    
    # Ensure cos_theta is within the valid range for arccos due to numerical issues
    data['cos_theta'] = data['cos_theta'].clip(-1, 1)
    
    data['theta'] = np.arccos(data['cos_theta'])
    #print(data['theta'].head(20))
    '''

    # Compute time differences in seconds
    data['time_difference'] = data['Timestamp'].diff().shift(-1) * SECONDS_PER_UNIT
    #print(data['time_difference'].head(20))
    
    # Calculate gaze velocity in radians per second, then convert to degrees per second
    data['gaze_velocity_radians'] = data['theta'] / data['time_difference']
    data['gaze_velocity'] = data['gaze_velocity_radians'] * RADIANS_TO_DEGREES
    #print(data['gaze_velocity'].head(20))
    
    # Mask out values where gaze_velocity is greater than 800 or less than or equal to 0
    data.loc[(data['gaze_velocity'] > 800), 'gaze_velocity'] = np.nan
    
    # Interpolate missing values that were masked out
    data['gaze_velocity_smoothed'] = data['gaze_velocity'].interpolate()
    
    # Calculate the smoothed gaze velocity using a rolling window
    #data['gaze_velocity_smoothed'] = data['gaze_velocity'].rolling(window=2).median()
    
    # Drop rows where averaging could not be applied (e.g., start and end of the dataset)
    data = data.dropna(subset=['gaze_velocity'])
    
    # Detect transitions where the target changes from True to False or vice versa
    data['transition'] = data['Target'].ne(data['Target'].shift())
    transition_times = data[data['transition']].index
    
    # Split the transformed data into individual components
    data[['gazeOrigin_x', 'gazeOrigin_y', 'gazeOrigin_z', 'gazeOrigin_w']] = pd.DataFrame(data['world_gaze_origin'].tolist(), index=data.index)
    data[['gazeDirection_x', 'gazeDirection_y', 'gazeDirection_z']] = pd.DataFrame(data['world_gaze_direction'].tolist(), index=data.index)
    
    data['eyeDataTimestamp'] = data['Timestamp']

    '''
    # Plot gaze velocity
    plt.figure()
    plt.plot(data.index, data['gaze_velocity'], label='Gaze Velocity')
    plt.plot(data.index, data['gaze_velocity_smoothed'], label='Smoothed Gaze Velocity', color='red')
    for t in transition_times:
        plt.axvline(x=t, color='blue', linestyle=':', linewidth=1.5)  # Start of transition
    plt.xlabel('Sample')
    plt.ylabel('Gaze Velocity (degrees per second)')
    plt.legend()
    plt.title('Gaze Velocity Over Samples')
    plt.show()
    '''
    
    # Create a new DataFrame or modify an existing one
    features = features.assign(Timestamp = data['Timestamp'], time_difference=data['time_difference'], gaze_velocity=data['gaze_velocity'])
    
    # Detect fixations based on the IDT algorithm
    fixation_events = list(detect_fixations(data))

    if len(fixation_events)>0:
        df_fixations = pd.DataFrame(fixation_events)
        
        # Assuming df_fixations already includes columns for 'start_timestamp', 'duration', and unique fixation identifiers
        df_fixations['fixation_event'] = np.arange(len(df_fixations)) + 1  # Assign unique identifier for each fixation
        
        # Sort the fixation DataFrame by 'start_timestamp' for proper asof merging
        df_fixations.sort_values('fixation_start', inplace=True)
        
        # Merge using 'pd.merge_asof' with backward direction to capture ongoing fixations
        data = pd.merge_asof(data.sort_values('eyeDataTimestamp'), df_fixations,
                             left_on='eyeDataTimestamp', right_on='fixation_start',
                             direction='backward')
    
        # Check both start and end conditions to confirm fixation presence
        data['is_fixation'] = (data['eyeDataTimestamp'] >= data['fixation_start']) & (data['eyeDataTimestamp'] <= data['fixation_end'])
        data['fixation_event'] = data['fixation_event'].where(data['is_fixation'], 0)
        data['fixation_duration'] = data['fixation_duration'].where(data['is_fixation'], 0)
        data['fixation_dispersion'] = data['fixation_dispersion'].where(data['is_fixation'], 0)
        data['fixation_centroid'] = data['fixation_centroid'].where(data['is_fixation'], 0)
        # Create a new column 'is_fixation_int' for the integer representation of 'is_fixation'
        data['is_fixation_int'] = data['is_fixation'].astype(int)
    else:
        # If no fixation events, add default columns with zeros
        data['is_fixation'] = False
        data['fixation_event'] = 0
        data['fixation_duration'] = 0
        data['fixation_dispersion'] = 0
        data['fixation_centroid'] = 0
        data['is_fixation_int'] = 0
    
    # Resultant DataFrame 'merged_data' contains the original data with fixation flags and events updated
    #print("Processed data with fixation detection saved successfully.")
    
    # Create a new DataFrame or modify an existing one
    #features = features.assign(is_fixation=data['is_fixation'], fixation_event=data['fixation_event'],
     #                         fixation_duration=data['fixation_duration'], fixation_dispersion=data['fixation_dispersion'],
      #                        fixation_centroid=data['fixation_centroid'])
    
    # Create a new DataFrame or modify an existing one
    features = pd.merge(features, 
                        data[['Timestamp', 'is_fixation_int', 'fixation_event', 'fixation_duration', 'fixation_dispersion']],
                        on='Timestamp', 
                        how='left')

    # Detecting saccades based on velocity threshold
    data['is_saccade'] = data['gaze_velocity'] > SACCADE_VELOCITY_THRESHOLD
    
    # Create an event marker for changes in is_saccade status
    data['saccade_changes'] = data['is_saccade'].astype(int).diff().fillna(1).astype(bool)
    
    # Assign event numbers to saccades using cumsum method
    data['saccade_event'] = (data['saccade_changes'] & data['is_saccade']).cumsum()
    
    # Ensure that saccade_event is 0 where is_saccade is False
    data.loc[~data['is_saccade'], 'saccade_event'] = 0

    # Filter saccades by duration constraints
    saccade_groups = data.groupby('saccade_event')
    saccade_durations = saccade_groups['time_difference'].transform('sum')
    valid_saccades = (saccade_durations >= MIN_SACCADE_DURATION) & (saccade_durations <= MAX_SACCADE_DURATION)
    
    # Adjust is_saccade based on valid durations and reset saccade_event where not valid
    data['is_saccade'] = valid_saccades & data['is_saccade']
    
    # Setting saccade_changes and saccade_event
    data['saccade_changes'] = data['is_saccade'].astype(int).diff().fillna(1).astype(bool)
    data['saccade_event'] = (data['saccade_changes'] & data['is_saccade']).cumsum()
    data.loc[~data['is_saccade'], 'saccade_event'] = 0
    
    # Create a new column 'is_fixation_int' for the integer representation of 'is_fixation'
    data['is_saccade_int'] = data['is_saccade'].astype(int)
    
    # Ensuring the column type is float to accommodate NaN and inf
    data['saccade_duration'] = 0.0
    data.loc[data['is_saccade'], 'saccade_duration'] = saccade_groups['time_difference'].transform('sum').fillna(0).astype(float)
    
    # Assuming dispersion is calculated and might have inf or NaN, set dtype to float
    data['saccade_dispersion'] = 0.0
    for event_id, group in data[data['is_saccade']].groupby('saccade_event'):
        dispersion = gaze_dispersion(group.to_dict('records'))
        data.loc[group.index, 'saccade_dispersion'] = np.nan_to_num(dispersion, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
    
    # Calculate acceleration
    data['gaze_acceleration'] = data['gaze_velocity'].diff() / data['time_difference'].replace(0, pd.NA)
    
    #print("Processed data with saccade features saved successfully.")

    # Create a new DataFrame or modify an existing one
    features = pd.merge(features, 
                        data[['Timestamp', 'is_saccade_int', 'saccade_event', 'saccade_duration', 'saccade_dispersion', 'gaze_acceleration']],
                        on='Timestamp', 
                        how='left')
    
    # Detecting saccades and fixations based on smoothed gaze velocity
    data['event'] = np.where(data['is_saccade'], 'saccade',
                             np.where(data['is_fixation'], 'fixation', 'None'))
    # Check distribution of events
    #print("Event distribution:", data['event'].value_counts())
    
    # Create a new DataFrame or modify an existing one
    features = features.assign(event=data['event'])

    compute_feature(data)
    
    data['fixation_x_mean'] = data['fixation_x_mean'].where(data['is_fixation'], 0)
    data['fixation_x_std'] = data['fixation_x_std'].where(data['is_fixation'], 0)
    data['fixation_y_mean'] = data['fixation_y_mean'].where(data['is_fixation'], 0)
    data['fixation_y_std'] = data['fixation_y_std'].where(data['is_fixation'], 0)
    data['saccade_x_mean'] = data['saccade_x_mean'].where(data['is_saccade'], 0)
    data['saccade_x_std'] = data['saccade_x_std'].where(data['is_saccade'], 0)
    data['saccade_y_mean'] = data['saccade_y_mean'].where(data['is_saccade'], 0)
    data['saccade_y_std'] = data['saccade_y_std'].where(data['is_saccade'], 0)
    data['fixation_x_skew'] = data['fixation_x_skew'].where(data['is_fixation'], 0)
    data['fixation_y_skew'] = data['fixation_y_skew'].where(data['is_fixation'], 0)
    data['fixation_x_kurt'] = data['fixation_x_kurt'].where(data['is_fixation'], 0)
    data['fixation_y_kurt'] = data['fixation_y_kurt'].where(data['is_fixation'], 0)
    data['saccade_amplitude'] = data['saccade_amplitude'].where(data['is_saccade'], 0)
    #data['k_coefficient'] = data['k_coefficient'].where(data['is_fixation'], 0)
    data['fixation_path_length'] = data['fixation_path_length'].where(data['is_fixation'], 0)
    data['saccade_path_length'] = data['saccade_path_length'].where(data['is_saccade'], 0)
    data['fixation_angular_displacement_centroid'] = data['fixation_angular_displacement_centroid'].where(data['is_fixation'], 0)
    data['angular_displacement_last_sample'] = data['angular_displacement_last_sample'].where(data['is_fixation'], 0)
    data['saccadic_ratio'] = data['saccadic_ratio'].where(data['is_saccade'], 0)
    data['angular_displacement_saccade_point'] = data['angular_displacement_saccade_point'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_mean'] = data['saccade_gaze_velocity_mean'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_median'] = data['saccade_gaze_velocity_median'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_max'] = data['saccade_gaze_velocity_max'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_std_dev'] = data['saccade_gaze_velocity_std_dev'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_skew'] = data['saccade_gaze_velocity_skew'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_kurtosis'] = data['saccade_gaze_velocity_kurtosis'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_mean'] = data['saccade_gaze_acceleration_mean'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_median'] = data['saccade_gaze_acceleration_median'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_max'] = data['saccade_gaze_acceleration_max'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_std_dev'] = data['saccade_gaze_acceleration_std_dev'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_skew'] = data['saccade_gaze_acceleration_skew'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_kurtosis'] = data['saccade_gaze_acceleration_kurtosis'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_x_mean'] = data['saccade_gaze_velocity_x_mean'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_x_median'] = data['saccade_gaze_velocity_x_median'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_x_max'] = data['saccade_gaze_velocity_x_max'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_x_std_dev'] = data['saccade_gaze_velocity_x_std_dev'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_x_skew'] = data['saccade_gaze_velocity_x_skew'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_x_kurtosis'] = data['saccade_gaze_velocity_x_kurtosis'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_y_mean'] = data['saccade_gaze_velocity_y_mean'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_y_median'] = data['saccade_gaze_velocity_y_median'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_y_max'] = data['saccade_gaze_velocity_y_max'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_y_std_dev'] = data['saccade_gaze_velocity_y_std_dev'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_y_skew'] = data['saccade_gaze_velocity_y_skew'].where(data['is_saccade'], 0)
    data['saccade_gaze_velocity_y_kurtosis'] = data['saccade_gaze_velocity_y_kurtosis'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_x_mean'] = data['saccade_gaze_acceleration_x_mean'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_x_median'] = data['saccade_gaze_acceleration_x_median'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_x_max'] = data['saccade_gaze_acceleration_x_max'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_x_std_dev'] = data['saccade_gaze_acceleration_x_std_dev'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_x_skew'] = data['saccade_gaze_acceleration_x_skew'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_x_kurtosis'] = data['saccade_gaze_acceleration_x_kurtosis'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_y_mean'] = data['saccade_gaze_acceleration_y_mean'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_y_median'] = data['saccade_gaze_acceleration_y_median'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_y_max'] = data['saccade_gaze_acceleration_y_max'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_y_std_dev'] = data['saccade_gaze_acceleration_y_std_dev'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_y_skew'] = data['saccade_gaze_acceleration_y_skew'].where(data['is_saccade'], 0)
    data['saccade_gaze_acceleration_y_kurtosis'] = data['saccade_gaze_acceleration_y_kurtosis'].where(data['is_saccade'], 0)
    data['fixation_velocity_mean'] = data['fixation_velocity_mean'].where(data['is_fixation'], 0)

    # Create a new DataFrame or modify an existing one
    
    features = pd.merge(features, 
                        data[['Timestamp', 'fixation_x_mean', 'fixation_x_std', 
                              'fixation_y_mean', 'fixation_y_std', 'saccade_x_mean', 'saccade_x_std',
                             'saccade_y_mean', 'saccade_y_std', 'fixation_x_skew', 'fixation_y_skew',
                             'fixation_x_kurt', 'fixation_y_kurt', 'saccade_amplitude', 'k_coefficient',
                             'fixation_path_length', 'saccade_path_length', 'fixation_angular_displacement_centroid',
                             'angular_displacement_last_sample', 'saccadic_ratio', 'angular_displacement_saccade_point',
                             'saccade_gaze_velocity_mean', 'saccade_gaze_velocity_median', 'saccade_gaze_velocity_max',
                             'saccade_gaze_velocity_std_dev', 'saccade_gaze_velocity_skew', 'saccade_gaze_velocity_kurtosis',
                             'saccade_gaze_acceleration_mean', 'saccade_gaze_acceleration_median', 'saccade_gaze_acceleration_max',
                             'saccade_gaze_acceleration_std_dev', 'saccade_gaze_acceleration_skew', 'saccade_gaze_acceleration_kurtosis',
                             'saccade_gaze_velocity_x_mean', 'saccade_gaze_velocity_x_median', 'saccade_gaze_velocity_x_max',
                             'saccade_gaze_velocity_x_std_dev', 'saccade_gaze_velocity_x_skew', 'saccade_gaze_velocity_x_kurtosis',
                             'saccade_gaze_velocity_y_mean', 'saccade_gaze_velocity_y_median', 'saccade_gaze_velocity_y_max',
                             'saccade_gaze_velocity_y_std_dev', 'saccade_gaze_velocity_x_skew', 'saccade_gaze_velocity_y_kurtosis',
                             'saccade_gaze_acceleration_x_mean', 'saccade_gaze_acceleration_x_median', 'saccade_gaze_acceleration_x_max',
                             'saccade_gaze_acceleration_x_std_dev', 'saccade_gaze_acceleration_x_skew', 'saccade_gaze_acceleration_x_kurtosis',
                             'saccade_gaze_acceleration_y_mean', 'saccade_gaze_acceleration_y_median', 'saccade_gaze_acceleration_y_max',
                             'saccade_gaze_acceleration_y_std_dev', 'saccade_gaze_acceleration_y_skew', 'saccade_gaze_acceleration_y_kurtosis',
                             'fixation_velocity_mean']],
                        on='Timestamp', 
                        how='left')

    # Add a column for dispersion
    data['dispersion'] = 0.0
    
    '''
    # Calculate dispersion for each sample within one second window
    for index, row in data.iterrows():
        start_time = data['Timestamp'].min()
        end_time = data['Timestamp'].max()
        
        window_data = data[(data['Timestamp'] >= start_time) & (data['Timestamp'] <= end_time)]
        formatted_data = window_data.to_dict('records')
        
        data.loc[index, 'dispersion'] = modified_gaze_dispersion(formatted_data)
    '''
        
    start_time = data['Timestamp'].min()
    end_time = data['Timestamp'].max()
        
    window_data = data[(data['Timestamp'] >= start_time) & (data['Timestamp'] <= end_time)]
    formatted_data = window_data.to_dict('records')
        
    data.loc['dispersion'] = modified_gaze_dispersion(formatted_data)
    
    # Create a new DataFrame or modify an existing one
    #features = features.assign(dispersion=data['dispersion'])
    
    # Create a new DataFrame or modify an existing one
    features = pd.merge(features, 
                        data[['Timestamp', 'dispersion']],
                        on='Timestamp', 
                        how='left')
    
    return features