import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV data into a DataFrame
file_path = 'hand_joint_data.csv'
data_df = pd.read_csv(file_path)

# Convert timestamp to seconds (from 100 ns units)
data_df['Timestamp'] = data_df['Timestamp'] * 1e-7  # 100 ns = 1e-7 seconds

# Function to calculate velocity and acceleration
def calculate_velocity_and_acceleration(df, joint_name):
    # Filter data for the specific joint and only for the "Right" hand
    joint_data = df[(df['Joint'] == joint_name) & (df['Hand'] == 'Right')]
    
    # Ensure data is sorted by Timestamp
    joint_data = joint_data.sort_values('Timestamp').reset_index(drop=True)
    
    # Calculate the position differences
    positions = joint_data[['Position_X', 'Position_Y', 'Position_Z']].values
    timestamps = joint_data['Timestamp'].values
    
    # Calculate velocity (delta_position / delta_time)
    delta_times = np.diff(timestamps)
    print(delta_times[:20])
    velocities = np.diff(positions, axis=0) / delta_times[:, None]
    
    # Calculate acceleration (delta_velocity / delta_time)
    delta_times_acc = delta_times[1:]
    accelerations = np.diff(velocities, axis=0) / delta_times_acc[:, None]
    
    return velocities, accelerations, joint_data

# Define the joint name to analyze (can be changed as needed)
joint_name = 'Wrist'

# Joint names for clarity
joint_names = [
    "Palm", "Wrist", "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip"
]

for joint_name in joint_names:

    # Calculate velocity and acceleration for the selected joint
    velocities, accelerations, joint_data = calculate_velocity_and_acceleration(data_df, joint_name)
    
    # Plotting the velocities and accelerations
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot velocity
    axes[0].plot(joint_data['Timestamp'][1:], np.linalg.norm(velocities, axis=1), label='Velocity Magnitude')
    axes[0].set_title(f'{joint_name} - Velocity Magnitude')
    axes[0].set_xlabel('Timestamp')
    axes[0].set_ylabel('Velocity')
    axes[0].legend()
    
    # Plot acceleration
    axes[1].plot(joint_data['Timestamp'][2:], np.linalg.norm(accelerations, axis=1), label='Acceleration Magnitude', color='orange')
    axes[1].set_title(f'{joint_name} - Acceleration Magnitude')
    axes[1].set_xlabel('Timestamp')
    axes[1].set_ylabel('Acceleration')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
