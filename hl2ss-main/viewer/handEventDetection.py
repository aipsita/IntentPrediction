import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV data into a DataFrame
file_path = 'hand_joint_data.csv'
data_df = pd.read_csv(file_path)

# Convert timestamp to seconds (from 100 ns units)
data_df['Timestamp'] = data_df['Timestamp'] * 1e-7  # 100 ns = 1e-7 seconds

# Joints to consider for finger movement detection
top_finger_joints = [
    'IndexTip'  # Add other joints as needed: 'MiddleTip', 'RingTip', 'LittleTip'
]

# Threshold for detecting significant acceleration changes
ACCELERATION_THRESHOLD = 20  # units/s²

# Determine the color based on the maximum absolute acceleration component
def determine_color(acceleration):
    # Calculate the absolute values of the components
    abs_accel = np.abs(acceleration)
    
    # Find the index of the maximum absolute value
    max_index = np.argmax(abs_accel)
    
    # Determine the sign of the maximum component
    max_value = acceleration[max_index]
    
    # Set color based on the index and sign of the maximum component
    if abs(max_value) < ACCELERATION_THRESHOLD:
        return 'black'  # No significant acceleration
    
    if max_index == 0:  # X component is largest
        color = 'red' if max_value > 0 else 'blue'
    elif max_index == 1:  # Y component is largest
        color = 'green' if max_value > 0 else 'purple'
    else:  # Z component is largest
        color = 'orange' if max_value > 0 else 'cyan'
    
    return color

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
    
    velocities = np.diff(positions, axis=0) / delta_times[:, None]
    
    # Calculate acceleration (delta_velocity / delta_time)
    delta_times_acc = delta_times[1:]
    
    accelerations = np.diff(velocities, axis=0) / delta_times_acc[:, None]
    
    # Check for zero, infinite, or NaN values in accelerations and count them
    zero_accelerations_count = np.sum(np.all(accelerations == 0, axis=1))
    inf_accelerations_count = np.sum(np.isinf(accelerations))
    nan_accelerations_count = np.sum(np.isnan(accelerations))
    
    print(f"Joint: {joint_name}")
    print(f"Zero acceleration vectors: {zero_accelerations_count}")
    print(f"Infinite acceleration vectors: {inf_accelerations_count}")
    print(f"NaN acceleration vectors: {nan_accelerations_count}")
    
    return velocities, accelerations, joint_data

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Analyze the specified finger joints for significant acceleration events
for joint_name in top_finger_joints:
    # Calculate velocity and acceleration for the selected joint
    velocities, accelerations, joint_data = calculate_velocity_and_acceleration(data_df, joint_name)

    # Calculate acceleration magnitude
    acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)

    # Determine dominant direction and plot accordingly
    num_frames = len(acceleration_magnitudes)
    for start in range(0, num_frames, 10):
        end = min(start + 10, num_frames)
        segment_accelerations = accelerations[start:end]
        segment_timestamps = joint_data['Timestamp'][start + 2:end + 2]

        # Determine the most frequent significant direction
        colors = [determine_color(accel) for accel in segment_accelerations]
        significant_colors = [color for color in colors if color != 'black']
        if significant_colors:
            # Count occurrences of each color
            color_counts = {color: significant_colors.count(color) for color in set(significant_colors)}
            dominant_color = max(color_counts, key=color_counts.get)
        else:
            dominant_color = 'black'  # Default if no significant event

        # Plot the segment
        ax.plot(segment_timestamps, acceleration_magnitudes[start:end], color=dominant_color)

# Customize plot
ax.set_title('Acceleration Magnitudes with Dominant Direction per 10 Frames')
ax.set_xlabel('Timestamp (seconds)')
ax.set_ylabel('Acceleration (units/s²)')
ax.legend()

plt.tight_layout()
plt.show()
