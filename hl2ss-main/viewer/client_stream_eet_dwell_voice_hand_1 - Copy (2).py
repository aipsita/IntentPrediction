import numpy as np
import pandas as pd
import csv
from pynput import keyboard
import threading
import hl2ss
import hl2ss_lnm
import hl2ss_rus
import hl2ss_utilities
import time
import queue
import scipy.fftpack
import sys
import os
from scipy.ndimage import uniform_filter1d
import pyaudio

# Settings --------------------------------------------------------------------

# HoloLens address
host = "192.168.1.179"

# General Settings
fps = 90
position = [0, 0, 1]
rotation = [0, 0, 0, 1]
font_size = 0.4
rgba = [1, 1, 1, 1]
enable = True
timer = None
target_state = False  # Initialize the target state
key = 0  # Initialize key
data_points = []  # List to store 100 data points
is_selection = 0.0  # Variable to store selection prediction

# Hand Event Detection Settings
VELOCITY_THRESHOLD = 0.05  # units/s, adjust as needed
HEAD_VELOCITY_THRESHOLD = 0.01  # units/s, adjust as needed
ACCELERATION_THRESHOLD = 1  # units/s²
joint_indices = {
    'IndexIntermediate': 8,
    'IndexTip': 10,
    'MiddleTip': 15,
}

# Define priority levels (lower number means higher priority)
PRIORITY_HIGHEST = 1
PRIORITY_HIGH = 2
PRIORITY_NORMAL = 3

# Create a priority queue
#priority_queue = queue.PriorityQueue()

profile = hl2ss.AudioProfile.AAC_24000
audio_format = pyaudio.paInt16 if (profile == hl2ss.AudioProfile.RAW) else pyaudio.paFloat32

# Initialize Unity IPC
ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
ipc.open()

client_eye = hl2ss_lnm.rx_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, fps=fps)
client_voice = hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE, profile=profile)
client_hand = hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT)
#client_head = hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT)

# Add the sub_folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'EyeTracking'))
   
try:
    from ModelFixation import get_fixation
except ImportError as e:
    print(f"Error importing model testing")

def inform_connect():
    global timer, enable
    start_time = time.time()
    collect_data()
    end_time = time.time()
    #print(f"Time taken: {end_time - start_time}")
    if enable:
        schedule_function1()
        
def schedule_function1():
    global timer
    interval = 0.4  # 200 milliseconds
    
    if timer is not None:
        timer.cancel()  # Cancel the previous timer if it exists
    
    timer = threading.Timer(interval, inform_connect)
    timer.start()

def collect_data():
    global data_points, is_selection
    
    count = 0
    while count < 30:
        data = client_eye.get_next_packet()
        eet = hl2ss.unpack_eet(data.payload)

        # Create a dictionary for the data point
        data_point = {
            'Timestamp': data.timestamp,
            'Pose': data.pose,
            'Calibration_Valid': eet.calibration_valid,
            'Combined_Eye_Gaze_Valid': eet.combined_ray_valid,
            'Combined_Eye_Gaze_Origin': eet.combined_ray.origin,
            'Combined_Eye_Gaze_Direction': eet.combined_ray.direction,
            'Left_Eye_Gaze_Valid': eet.left_ray_valid,
            'Left_Eye_Gaze_Origin': eet.left_ray.origin,
            'Left_Eye_Gaze_Direction': eet.left_ray.direction,
            'Right_Eye_Gaze_Valid': eet.right_ray_valid,
            'Right_Eye_Gaze_Origin': eet.right_ray.origin,
            'Right_Eye_Gaze_Direction': eet.right_ray.direction,
            'Left_Eye_Openness_Valid': eet.left_openness_valid,
            'Left_Eye_Openness': eet.left_openness,
            'Right_Eye_Openness_Valid': eet.right_openness_valid,
            'Right_Eye_Openness': eet.right_openness,
            'Vergence_Distance_Valid': eet.vergence_distance_valid,
            'Vergence_Distance': eet.vergence_distance,
            'Target': target_state
        }

        if ((data_point['Calibration_Valid'] == True) & (data_point['Combined_Eye_Gaze_Valid'] == True)):
            data_points.append(data_point)
            count += 1
        
    # Convert the list of dictionaries to a DataFrame
    data_1 = pd.DataFrame(data_points)
    is_selection = get_fixation(data_1)
    data_points = []
    if is_selection > 0.5:
        send_signal_to_unity('Selection')
        print("Target Selection")
    else:
        send_signal_to_unity('NotSelection')

# Function to handle key press events
def on_press(key):
    global enable
    if key == keyboard.Key.esc:
        enable = False
        if timer:
            timer.cancel()
    return enable

# Function to manage data collection
def data_collection():
    global enable
    
    client_eye.open()
    try:
        # Initialize and start the timer
        schedule_function1()  # This starts the first timer
        while enable:
            time.sleep(0.1)  # Keep the thread alive
    finally:
        # Close IPC connection after all threads have completed
        ipc.close()
        client_eye.close()

def pcmworker(pcmqueue):
    global enable, audio_format
    p = pyaudio.PyAudio()
    stream = p.open(format=audio_format, channels=hl2ss.Parameters_MICROPHONE.CHANNELS, rate=hl2ss.Parameters_MICROPHONE.SAMPLE_RATE, output=True)
    stream.start_stream()
    while enable:
        stream.write(pcmqueue.get())
    stream.stop_stream()
    stream.close()

def is_speaking(audio_data, threshold=0.01):
    audio_signal = np.frombuffer(audio_data, dtype=np.float32)
    rms = np.sqrt(np.mean(audio_signal**2))
    return rms > threshold

def reduce_noise(audio_signal, spectral_threshold=0.02, volume_threshold=0.01):
    fft_signal = scipy.fftpack.fft(audio_signal)
    magnitude = np.abs(fft_signal)
    phase = np.angle(fft_signal)
    magnitude[magnitude < spectral_threshold] = 0
    fft_signal = magnitude * np.exp(1j * phase)
    audio_signal_clean = np.real(scipy.fftpack.ifft(fft_signal))
    rms = np.sqrt(np.mean(audio_signal_clean**2))
    if rms < volume_threshold:
        audio_signal_clean *= 0.5
    return audio_signal_clean

def send_signal_to_unity(message):
    buffer = hl2ss_rus.command_buffer()
    buffer.connect_success(message)
    ipc.push(buffer)
    response = ipc.pull(buffer)

# Function to manage audio recording
def audio_recording():
    global enable
    
    pcmqueue = queue.Queue()
    thread = threading.Thread(target=pcmworker, args=(pcmqueue,))
    thread.start()
    
    client_voice.open()
    
    while enable:
        data_voice = client_voice.get_next_packet()
        audio = hl2ss_utilities.microphone_planar_to_packed(data_voice.payload) if (profile != hl2ss.AudioProfile.RAW) else data_voice.payload
        
        audio_contiguous = np.ascontiguousarray(audio)
        audio_signal = np.frombuffer(audio_contiguous, dtype=np.float32)
        audio_clean = reduce_noise(audio_signal)
        
        pcmqueue.put(audio_clean.tobytes())
        audio_data = pcmqueue.get()
        #print("Here")
        
        if is_speaking(audio_data):
            print("User is speaking")
            send_signal_to_unity('UserSpeaking')
            
        data_voice = []

    client_voice.close()
    pcmqueue.put(b'')
    thread.join()

# Function to handle hand event detection
def hand_event_detection():
    global enable

    data = []
    data_head = []
    client_hand.open()

    # Use to append new data
    def append_data(timestamp, positions, index_intermediate, index_tip):
        data.append({'Timestamp': timestamp, 'Positions': positions, 
                     'IndexIntermediate': index_intermediate, 'IndexTip': index_tip})
        
    previous_dominant_direction = 'no significant direction'
    previous_head_dominant_direction = 'no significant direction'
    
    # Main loop
    while enable:
        packet = client_hand.get_next_packet()
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
            if len(data) >= 20:
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
                            
                            if dominant_direction == 'Positive X':
                                send_signal_to_unity("HandRight")
                                #send_signal_to_unity_priority('Right', priority=PRIORITY_HIGH)
                            if dominant_direction == 'Negative X':
                                send_signal_to_unity("HandLeft")
                                #send_signal_to_unity_priority('Left', priority=PRIORITY_HIGH)
                            if dominant_direction == 'Positive Y':
                                send_signal_to_unity("HandUp")
                                #send_signal_to_unity_priority('Up', priority=PRIORITY_HIGH)
                            if dominant_direction == 'Negative Y':
                                send_signal_to_unity("HandDown")
                                #send_signal_to_unity_priority('Down', priority=PRIORITY_HIGH)
                            if dominant_direction == 'Positive Z':
                                send_signal_to_unity("HandBackward")
                                #send_signal_to_unity_priority('Forward', priority=PRIORITY_HIGH)
                            if dominant_direction == 'Negative Z':
                                send_signal_to_unity("HandForward")
                                #send_signal_to_unity_priority('Backward', priority=PRIORITY_HIGH)
                        
                    previous_dominant_direction = dominant_direction
                
                # Clear the data for the next set of frames
                data = []
                
        elif si.is_valid_head_pose():
            head_pose = si.get_head_pose()
            head_position = head_pose.position  # Get head position
            data_head.append({'Timestamp': packet.timestamp, 'Position': head_position})
        
            # Process every 10 frames
            if len(data_head) >= 20:
                head_positions = np.array([item['Position'] for item in data_head])
                head_timestamps = np.array([item['Timestamp'] for item in data_head]) * 1e-7  # Convert timestamps to seconds
                
                # Smooth positions to reduce noise
                head_smoothed_positions = uniform_filter1d(head_positions, size=5, axis=0)
                
                # Calculate velocities
                head_delta_times = np.diff(head_timestamps)
                head_velocities = np.diff(head_smoothed_positions, axis=0) / head_delta_times[:, None]
                head_velocity_magnitudes = np.linalg.norm(head_velocities, axis=1)
                
                # Determine significant movements based on a velocity threshold
                head_significant_indices = head_velocity_magnitudes > HEAD_VELOCITY_THRESHOLD
                head_significant_velocities = head_velocities[head_significant_indices]
                
                if head_significant_velocities.any():
                    # Determine dominant direction
                    head_dominant_direction = determine_direction(np.mean(head_significant_velocities, axis=0))
                    if previous_head_dominant_direction == head_dominant_direction:
                        print(f"Dominant head movement direction for last 10 frames: {head_dominant_direction}")
                        
                        if head_dominant_direction == 'Positive X':
                            send_signal_to_unity("HeadRight")
                            #send_signal_to_unity_priority('Right', priority=PRIORITY_HIGH)
                        if head_dominant_direction == 'Negative X':
                            send_signal_to_unity("HeadLeft")
                            #send_signal_to_unity_priority('Left', priority=PRIORITY_HIGH)
                        if head_dominant_direction == 'Positive Y':
                            send_signal_to_unity("HeadUp")
                            #send_signal_to_unity_priority('Up', priority=PRIORITY_HIGH)
                        if head_dominant_direction == 'Negative Y':
                            send_signal_to_unity("HeadDown")
                            #send_signal_to_unity_priority('Down', priority=PRIORITY_HIGH)
                        if head_dominant_direction == 'Positive Z':
                            send_signal_to_unity("HeadBackward")
                            #send_signal_to_unity_priority('Forward', priority=PRIORITY_HIGH)
                        if head_dominant_direction == 'Negative Z':
                            send_signal_to_unity("HeadForward")
                            #send_signal_to_unity_priority('Backward', priority=PRIORITY_HIGH)  
                    previous_head_dominant_direction = head_dominant_direction
                
                data_head = []  # Clear data for next set of frames

    client_hand.close()

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

# Main function to run all functionalities
def main():
    global enable
    enable = True  # Ensure the enable flag is True at the start
    
    # Start the keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    # Start the threads
    data_thread = threading.Thread(target=data_collection)   
    audio_thread = threading.Thread(target=audio_recording)
    hand_thread = threading.Thread(target=hand_event_detection)
    #head_thread = threading.Thread(target=head_event_detection)
    
    data_thread.start()
    audio_thread.start()
    hand_thread.start()
    #head_thread.start()
    
    # Join the threads
    data_thread.join()
    audio_thread.join()
    hand_thread.join()
    #head_thread.join()
    
    # Stop the listener after all threads have completed
    listener.join()

# Execute the main function
main()
