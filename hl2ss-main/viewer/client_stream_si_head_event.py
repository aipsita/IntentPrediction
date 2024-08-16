import numpy as np
from scipy.ndimage import uniform_filter1d
from pynput import keyboard
import hl2ss
import hl2ss_lnm

# Settings
host = "192.168.1.179"  # HoloLens address
enable = True

# Real-time data storage
data = []

previous_dominant_direction = ""

# Thresholds
VELOCITY_THRESHOLD = 0.01  # units/s, adjust as needed

# Function to determine the dominant direction of movement
def determine_direction(vector):
    abs_vector = np.abs(vector)
    max_index = np.argmax(abs_vector)
    max_value = vector[max_index]
    directions = ['X', 'Y', 'Z']
    sign = 'Positive' if max_value > 0 else 'Negative'
    return f'{sign} {directions[max_index]}'

# Function to handle key press to terminate the program
def on_press(key):
    global enable
    if key == keyboard.Key.esc:
        enable = False
    return False

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT)
client.open()

# Main loop
while enable:
    packet = client.get_next_packet()
    si = hl2ss.unpack_si(packet.payload)
    
    if si.is_valid_head_pose():
        head_pose = si.get_head_pose()
        head_position = head_pose.position  # Get head position
        data.append({'Timestamp': packet.timestamp, 'Position': head_position})
    
    # Process every 10 frames
    if len(data) >= 20:
        positions = np.array([item['Position'] for item in data])
        timestamps = np.array([item['Timestamp'] for item in data]) * 1e-7  # Convert timestamps to seconds
        
        # Smooth positions to reduce noise
        smoothed_positions = uniform_filter1d(positions, size=5, axis=0)
        
        # Calculate velocities
        delta_times = np.diff(timestamps)
        velocities = np.diff(smoothed_positions, axis=0) / delta_times[:, None]
        velocity_magnitudes = np.linalg.norm(velocities, axis=1)
        
        # Determine significant movements based on a velocity threshold
        significant_indices = velocity_magnitudes > VELOCITY_THRESHOLD
        significant_velocities = velocities[significant_indices]
        
        if significant_velocities.any():
            # Determine dominant direction
            dominant_direction = determine_direction(np.mean(significant_velocities, axis=0))
            if(previous_dominant_direction == dominant_direction):
                print(f"Dominant head movement direction for last 10 frames: {dominant_direction}")
            previous_dominant_direction = dominant_direction
        
        data = []  # Clear data for next set of frames

client.close()
listener.stop()
