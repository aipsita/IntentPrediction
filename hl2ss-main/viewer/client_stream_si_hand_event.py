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
    'Wrist': 1,
    'IndexIntermediate': 8,
    'IndexTip': 10,
    'MiddleTip': 15,
    'ThumbTip': 5,
    'ThumbDistal': 4
    #'RingTip': 19,
    #'LittleTip': 24
}

# Real-time data storage
data = []

# Threshold for detecting significant movement
VELOCITY_THRESHOLD = 0.1  # units/s, adjust as needed
ACCELERATION_THRESHOLD = 1  # units/s²

previous_dominant_direction = 'no significant direction'

# Determine the direction based on the maximum absolute acceleration component
def determine_direction(acceleration):
    abs_accel = np.abs(acceleration)
    max_index = np.argmax(abs_accel)
    max_value = acceleration[max_index]
    
    if abs(max_value) < ACCELERATION_THRESHOLD:
        return 'no significant direction'  # No significant acceleration
    
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
def append_data(timestamp, positions, index_intermediate, index_tip, thumb_tip, thumb_distal):
    data.append({'Timestamp': timestamp, 'Positions': positions, 
                 'IndexIntermediate': index_intermediate, 'IndexTip': index_tip,
                 'ThumbTip': thumb_tip, 'ThumbDistal': thumb_distal})

# Main loop
while enable:
    packet = client.get_next_packet()
    si = hl2ss.unpack_si(packet.payload)
    
    if si.is_valid_hand_right():
        hand_right = si.get_hand_right()
        tip_positions = []
        index_intermediate_pose = hand_right.get_joint_pose(joint_indices['IndexIntermediate'])
        index_tip_pose = hand_right.get_joint_pose(joint_indices['IndexTip'])
        thumb_tip_pose = hand_right.get_joint_pose(joint_indices['ThumbTip'])
        thumb_distal_pose = hand_right.get_joint_pose(joint_indices['ThumbDistal'])
        
        wrist_pose = hand_right.get_joint_pose(1)
        for joint_name, joint_index in joint_indices.items():
            if joint_name != 'Wrist' and joint_name != 'IndexIntermediate' and joint_name != 'ThumbTip' and joint_name != 'ThumbDistal':  # Exclude metacarpal for the centroid calculation
                pose = hand_right.get_joint_pose(joint_index)
                tip_positions.append(pose.position - wrist_pose.position)
        
        if tip_positions:
            # Calculate centroid of the tip positions
            centroid = np.mean(tip_positions, axis=0)
            append_data(packet.timestamp, centroid, index_intermediate_pose.position, index_tip_pose.position, thumb_tip_pose.position, thumb_distal_pose.position)
        
        # Process every 10 frames
        if len(data) >= 10:
            # Extract centroids, timestamps, and index positions
            centroids = np.array([item['Positions'] for item in data])
            index_intermediates = np.array([item['IndexIntermediate'] for item in data])
            index_tips = np.array([item['IndexTip'] for item in data])
            thumb_distals = np.array([item['ThumbDistal'] for item in data])
            thumb_tips = np.array([item['ThumbTip'] for item in data])
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
            #print(velocity_magnitudes)
            significant_indices = velocity_magnitudes > VELOCITY_THRESHOLD
            
            # Determine the dominant direction from significant movements
            significant_accelerations = accelerations[significant_indices[:-1]]
            directions = [determine_direction(accel) for accel in significant_accelerations]
            #print(directions)
            significant_directions = [direction for direction in directions if direction != 'no significant direction']
            
            # Check direction based on majority of direction vectors
            if significant_directions:
                direction_counts = {direction: significant_directions.count(direction) for direction in set(significant_directions)}
                dominant_direction = max(direction_counts, key=direction_counts.get)
                
                # Calculate vector from index metacarpal to tip
                direction_vectors = index_tips - index_intermediates
                direction_vectors_thumb = thumb_tips - thumb_distals
                if "X" in dominant_direction:
                    dominant_axis = 0
                    expected_sign = 1 if "Positive" in dominant_direction else -1
                elif "Y" in dominant_direction:
                    dominant_axis = 1
                    expected_sign = 1 if "Positive" in dominant_direction else -1
                else:
                    dominant_axis = 2
                    expected_sign = 1 if "Positive" in dominant_direction else -1
                
                # Check if the majority of direction vectors match the dominant direction
                sign_matches = [
                    np.sign(direction_vectors[i, dominant_axis]) == expected_sign
                    for i in range(len(direction_vectors))
                ]
                
                sign_matches_thumb = [
                    np.sign(direction_vectors_thumb[i, dominant_axis]) == expected_sign
                    for i in range(len(direction_vectors_thumb))
                ]
                
                #print(sum(sign_matches))
                #print(len(sign_matches))               
                
                if sum(sign_matches) >= len(sign_matches) / 2:
                    #print(f"Dominant direction for last 10 frames: {dominant_direction}")
                    if previous_dominant_direction == dominant_direction:
                        print(f"Dominant direction for last 10 frames: {dominant_direction}")
                    previous_dominant_direction = dominant_direction
                    
                elif sum(sign_matches_thumb) >= len(sign_matches_thumb) / 2:
                    #print(f"Dominant direction for last 10 frames: {dominant_direction}")
                    if previous_dominant_direction == dominant_direction:
                        print(f"Dominant direction for last 10 frames thumb: {dominant_direction}")
                    previous_dominant_direction = dominant_direction
                '''
                else:
                    print("Detected direction is not significant based on metacarpal to tip vector.")
            else:
                print("No significant direction detected for last 10 frames.")
                '''
            
            # Clear the data for the next set of frames
            data = []

client.close()
listener.join()
