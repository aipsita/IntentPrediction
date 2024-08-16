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
host = "10.25.14.181"

# General Settings
fps = 90
position = [0, 0, 1]
rotation = [0, 0, 0, 1]
font_size = 0.4
rgba = [1, 1, 1, 1]
enable = True
running = True
timer = None
target_state = False  # Initialize the target state
key = 0  # Initialize key
data_points = []  # List to store 100 data points
is_selection = 0.0  # Variable to store selection prediction

# Hand Event Detection Settings
VELOCITY_THRESHOLD = 0.01  # units/s, adjust as needed
ACCELERATION_THRESHOLD = 5  # units/s²
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

# Add the sub_folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'EyeTracking'))
   
try:
    from ModelFixation import get_fixation
except ImportError as e:
    print(f"Error importing model testing")

def inform_connect():
    global value_to_check, timer
    start_time = time.time()
    '''
    buffer = hl2ss_rus.command_buffer()
    buffer.connect_success('Highlight')
    ipc.push(buffer)
    response = ipc.pull(buffer)
    if response[0] == 1:
        collect_data()
    '''
    if client_eye:
        collect_data()
    end_time = time.time()
    #print(f"Time taken: {end_time - start_time}")
    if running:
        timer = schedule_function1()
        
def schedule_function1():
    interval = 0.4  # 200 milliseconds
    timer = threading.Timer(interval, inform_connect)
    timer.start()
    return timer

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
    #dwellbuffer = hl2ss_rus.command_buffer() 
    if is_selection > 0.5:
        send_signal_to_unity('Selection')
        #send_signal_to_unity_priority('Selection', priority=PRIORITY_NORMAL)
        #dwellbuffer.connect_success('Selection')
        print("Target Selection")
    else:
        send_signal_to_unity('NotSelection')
        #send_signal_to_unity_priority('NotSelection', priority=PRIORITY_NORMAL)
        #dwellbuffer.connect_success('NotSelection')
        #collect_data()
    
    # Send command(s) in buffer to the Unity app and receive response (4 byte unsigned integer per command)
    #ipc.push(dwellbuffer)
    #response = ipc.pull(dwellbuffer)
    #print(response[0])

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
        timer = schedule_function1()  # Start the repeated function call
        while enable:
            time.sleep(0.1)  # Keep the thread alive
    finally:
        client_eye.close()

    #client_eye.close()
    #ipc.close()

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
    #print(f"Received response for {message}:")
    #print(response[0])

'''
# Function to manage sending signals with priority
def send_signal_to_unity_priority(message, priority=PRIORITY_NORMAL):
    priority_queue.put((priority, message))

def process_queue():
    while enable or not priority_queue.empty():
        try:
            # Get the highest priority item
            priority, message = priority_queue.get(timeout=0.1)
            buffer = hl2ss_rus.command_buffer()
            buffer.connect_success(message)
            ipc.push(buffer)
            response = ipc.pull(buffer)
            #print(f"Received response for {message}: {response[0]}")
        except queue.Empty:
            continue
'''

# Function to manage audio recording
def audio_recording():
    global enable
    
    pcmqueue = queue.Queue()
    thread = threading.Thread(target=pcmworker, args=(pcmqueue,))
    #listener = keyboard.Listener(on_press=on_press)
    thread.start()
    #listener.start()
    
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
            #send_signal_to_unity_priority('UserSpeaking', priority=PRIORITY_HIGHEST)
        #else:
            #send_signal_to_unity('NotSpeaking')
            #send_signal_to_unity_priority('NotSpeaking', priority=PRIORITY_HIGHEST)
            
        data_voice = []

    client_voice.close()
    pcmqueue.put(b'')
    thread.join()
    #listener.join()
    #ipc.close()

# Function to handle hand event detection
def hand_event_detection():
    global enable
    
    #listener = keyboard.Listener(on_press=on_press)
    #listener.start()

    data = []
    client_hand.open()

    def append_data(timestamp, positions, index_intermediate, index_tip):
        data.append({'Timestamp': timestamp, 'Positions': positions, 
                     'IndexIntermediate': index_intermediate, 'IndexTip': index_tip})

    while enable:
        packet = client_hand.get_next_packet()
        si = hl2ss.unpack_si(packet.payload)
        
        if si.is_valid_hand_right():
            hand_right = si.get_hand_right()
            tip_positions = []
            index_intermediate_pose = hand_right.get_joint_pose(joint_indices['IndexIntermediate'])
            index_tip_pose = hand_right.get_joint_pose(joint_indices['IndexTip'])
            
            for joint_name, joint_index in joint_indices.items():
                if joint_name != 'IndexIntermediate':  # Exclude metacarpal for the centroid calculation
                    pose = hand_right.get_joint_pose(joint_index)
                    tip_positions.append(pose.position)
            
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
                
                # Determine the dominant direction from significant movements
                significant_accelerations = accelerations[significant_indices[:-1]]
                directions = [determine_direction(accel) for accel in significant_accelerations]
                significant_directions = [direction for direction in directions if direction != 'no significant direction']
                
                # Check direction based on majority of direction vectors
                if significant_directions:
                    direction_counts = {direction: significant_directions.count(direction) for direction in set(significant_directions)}
                    dominant_direction = max(direction_counts, key=direction_counts.get)
                    
                    # Calculate vector from index metacarpal to tip
                    direction_vectors = index_tips - index_intermediates
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
                    
                    if sum(sign_matches) > len(sign_matches) / 2:
                        print(f"Dominant direction for last 10 frames: {dominant_direction}")
                        '''
                        if dominant_direction == 'Positive X':
                            send_signal_to_unity("Right")
                            #send_signal_to_unity_priority('Right', priority=PRIORITY_HIGH)
                        if dominant_direction == 'Negative X':
                            send_signal_to_unity("Left")
                            #send_signal_to_unity_priority('Left', priority=PRIORITY_HIGH)
                        if dominant_direction == 'Positive Y':
                            send_signal_to_unity("Up")
                            #send_signal_to_unity_priority('Up', priority=PRIORITY_HIGH)
                        if dominant_direction == 'Negative Y':
                            send_signal_to_unity("Down")
                            #send_signal_to_unity_priority('Down', priority=PRIORITY_HIGH)
                        if dominant_direction == 'Positive Z':
                            send_signal_to_unity("Forward")
                            #send_signal_to_unity_priority('Forward', priority=PRIORITY_HIGH)
                        if dominant_direction == 'Negative Z':
                            send_signal_to_unity("Backward")
                            #send_signal_to_unity_priority('Backward', priority=PRIORITY_HIGH)
                            '''
                
                # Clear the data for the next set of frames
                data = []

    client_hand.close()
    #listener.join()

# Function to determine direction based on acceleration
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
        return 'Negative Z' if max_value > 0 else 'Positive Z'

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
    #queue_thread = threading.Thread(target=process_queue)
    
    data_thread.start()
    audio_thread.start()
    hand_thread.start()
    #queue_thread.start()
    
    # Join the threads
    data_thread.join()
    audio_thread.join()
    hand_thread.join()
    
    # Close IPC connection after all threads have completed
    ipc.close()
    
    # Stop the listener after all threads have completed
    listener.join()

# Execute the main function
main()
