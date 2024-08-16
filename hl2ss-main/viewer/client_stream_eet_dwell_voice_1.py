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


host = '10.25.14.181'
fps = 90
position = [0, 0, 1]
rotation = [0, 0, 0, 1]
font_size = 0.4
rgba = [1, 1, 1, 1]
enable = True
running = True
timer = None

enable = True
target_state = False  # Initialize the target state

key = 0  # Initialize key

data_points = []  # List to store 100 data points
is_selection = 0.0 # Variable to store selection prediction

# This value will be checked by function1
value_to_check = False

profile = hl2ss.AudioProfile.AAC_24000

# Initialize Unity IPC
ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
ipc.open()

client_eye = hl2ss_lnm.rx_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, fps=fps)
client_voice = hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE, profile=profile)

# Add the sub_folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'EyeTracking'))
   
try:
   #from ModelTesting import main_model_testing  # Import the function
   from ModelFixation import get_fixation
   #print("Successfully imported process_data from test2.py")
except ImportError as e:
   print(f"Error importing model testing")

def inform_connect():
    global value_to_check, timer
    start_time = time.time()
    buffer = hl2ss_rus.command_buffer()
    buffer.connect_success('Highlight')
    ipc.push(buffer)
    response = ipc.pull(buffer)
    if response[0] == 1:
        collect_data()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
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
    dwellbuffer = hl2ss_rus.command_buffer() 
    if is_selection > 0.5:
        dwellbuffer.connect_success('Selection')
    else:
        dwellbuffer.connect_success('NotSelection')
        #collect_data()
    
    # Send command(s) in buffer to the Unity app and receive response (4 byte unsigned integer per command)
    ipc.push(dwellbuffer)
    response = ipc.pull(dwellbuffer)
    print(response[0])

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

# Function to manage data collection
def data_collection():
    global enable

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    client_eye.open()
    
    # Start the repeated function call
    timer = schedule_function1()

    listener.join()
    ipc.close()
    client_eye.close()
    #ipc.close()

def pcmworker(pcmqueue):
    while enable:
        audio_data = pcmqueue.get()
        if not audio_data:
            break
        # Process the audio data as needed

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
    enable = True
    pcmqueue = queue.Queue()

    client_voice.open()
    thread = threading.Thread(target=pcmworker, args=(pcmqueue,))
    thread.start()

    while enable:
        data = client_voice.get_next_packet()
        audio = hl2ss_utilities.microphone_planar_to_packed(data.payload) if (profile != hl2ss.AudioProfile.RAW) else data.payload
        
        audio_contiguous = np.ascontiguousarray(audio)
        audio_signal = np.frombuffer(audio_contiguous, dtype=np.float32)
        audio_clean = reduce_noise(audio_signal)
        
        pcmqueue.put(audio_clean.tobytes())
        audio_data = pcmqueue.get()
        #print("Here")
        if is_speaking(audio_data):
            print("User is speaking")
            send_signal_to_unity('UserSpeaking')
        else:
            send_signal_to_unity('NotSpeaking')

    client_voice.close()
    enable = False
    pcmqueue.put(b'')
    thread.join()

# Main function to run both threads
def main():
    data_thread = threading.Thread(target=data_collection)
    audio_thread = threading.Thread(target=audio_recording)
    data_thread.start()
    audio_thread.start()
    data_thread.join()
    audio_thread.join()

# Execute the main function
main()
