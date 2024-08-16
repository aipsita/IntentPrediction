import numpy as np
from scipy.ndimage import uniform_filter1d
from pynput import keyboard
import hl2ss
import hl2ss_lnm

# Settings --------------------------------------------------------------------

# HoloLens address
host = "192.168.1.179"

# Initialize variables
enable = True

# Joint indices
joint_indices = {
    'IndexIntermediate': 8,
    'IndexTip': 10,
    'MiddleTip': 15
}

# Real-time data storage
data = []

# Threshold for detecting significant movement
VELOCITY_THRESHOLD = 0.05  # units/s, adjust as needed
ACCELERATION_THRESHOLD = 1  # units/s²

previous_dominant_direction = 'no significant direction'

# Determine the direction based on the maximum absolute value component
def determine_direction(vector):
    abs_vector = np.abs(vector)
    max_index = np.argmax(abs_vector)
    max_value = vector[max_index]
    
    if max_index == 0:  # X component is largest
        return 'Positive X' if max_value > 0 else 'Negative X'
    elif max_index == 1:  # Y component is largest
        return 'Positive Y' if max_value > 0 else 'Negative Y'
    else:  # Z component is largest
        return 'Positive Z' if max_value > 0 else 'Negative Z'

# Function to handle key press
def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT)
client.open()

# Use to append new data
def append_data(timestamp, positions, index_intermediate, index_tip):
    data.append({'Timestamp': timestamp, 'Positions': positions, 
                 'IndexIntermediate': index_intermediate, 'IndexTip': index_tip})

# Main loop
while enable:
    packet = client.get_next_packet()
    si = hl2ss.unpack_si(packet.payload)
    
    if si.is_valid_hand_right():
        hand_right = si.get_hand_right()
        tip_positions = []
        index_intermediate_pose = hand_right.get_joint_pose(joint_indices['IndexIntermediate'])
        index_tip_pose = hand_right.get_joint_pose(joint_indices['IndexTip'])
        
        wrist_pose = hand_right.get_joint_pose(1)
        for joint_name, joint_index in joint_indices.items():
            if joint_name not in ['IndexIntermediate']:
                pose = hand_right.get_joint_pose(joint_index)
                tip_positions.append(pose.position - wrist_pose.position)
        
        if tip_positions:
            # Calculate centroid of the tip positions
            centroid = np.mean(tip_positions, axis=0)
            append_data(packet.timestamp, centroid, index_intermediate_pose.position, index_tip_pose.position)
        
        # Process every 10 frames
        if len(data) >= 10:
            # Extract centroids, timestamps, and index positions
            centroids = np.array([item['Positions'] for item in data])
            index_intermediates = np.array([item['IndexIntermediate'] for item in data])
            index_tips = np.array([item['IndexTip'] for item in data])
            timestamps = np.array([item['Timestamp'] for item in data]) * 1e-7  # Convert to seconds
            
            # Apply smoothing to the centroids to reduce noise from minor movements
            smoothed_centroids = uniform_filter1d(centroids, size=5, axis=0)
            
            # Calculate velocities and accelerations
            delta_times = np.diff(timestamps)
            velocities = np.diff(smoothed_centroids, axis=0) / delta_times[:, None]
            delta_times_acc = delta_times[1:]
            accelerations = np.diff(velocities, axis=0) / delta_times_acc[:, None]
            
            # Apply velocity threshold to ignore minor swaying movements
            velocity_magnitudes = np.linalg.norm(velocities, axis=1)
            significant_indices = velocity_magnitudes > VELOCITY_THRESHOLD
            
            # Filter out only significant velocities
            significant_velocities = velocities[significant_indices]
            significant_direction_vectors = index_tips[1:][significant_indices] - index_intermediates[1:][significant_indices]
            
            # Determine directions based on significant movements
            directions = [determine_direction(vec) for vec in significant_direction_vectors]
            
            significant_directions = [direction for direction in directions if direction != 'no significant direction']
            
            if significant_directions:
                direction_counts = {direction: significant_directions.count(direction) for direction in set(significant_directions)}
                max_direction_count = max(direction_counts.values())
                
                if max_direction_count > len(significant_directions) / 2:
                    dominant_direction = max(direction_counts, key=direction_counts.get)
                    if previous_dominant_direction == dominant_direction:
                        print(f"Dominant direction for last 10 frames: {dominant_direction}")
                            
                else:
                    dominant_direction = 'no significant direction'
                    
                previous_dominant_direction = dominant_direction
            
            # Clear the data for the next set of frames
            data = []

client.close()
listener.join()
