import numpy as np
import pandas as pd
import csv
from pynput import keyboard
import threading
import hl2ss
import hl2ss_lnm
import hl2ss_rus
import time
import threading
import sys
import os

# Settings --------------------------------------------------------------------

# HoloLens address
host = '10.25.244.47'

# Target Frame Rate
# Options: 30, 60, 90
fps = 90

# Position in world space (x, y, z) in meters
position = [0, 0, 1]

# Rotation in world space (x, y, z, w) as a quaternion
rotation = [0, 0, 0, 1]

# Font size
font_size = 0.4

# Text color
rgba = [1, 1, 1, 1]

#------------------------------------------------------------------------------

enable = True
target_state = False  # Initialize the target state

key = 0  # Initialize key

data_points = []  # List to store 100 data points
is_selection = 0.0 # Variable to store selection prediction

# This value will be checked by function1
value_to_check = False
running = True
timer = None
    
# Message functions -----------------------------------------------------------
# Inform server about succuessful connect
def inform_connect():
    global value_to_check, timer
    #threading.Timer(0.02, inform_connect).start()
    # Create command buffer and append command(s)
    buffer = hl2ss_rus.command_buffer() 
    buffer.connect_success('Highlight')
    
    # Send command(s) in buffer to the Unity app and receive response (4 byte unsigned integer per command)
    ipc.push(buffer)
    response = ipc.pull(buffer)
    
    # Evaluate response
    print("Received response for Highlight:")
    print(response[0])
    
    # Check the value
    if (response[0] == 1): #value_to_check:
        # If the value is True, run function2
        collect_data()

    # Schedule the next check if still running
    if running:
        timer = schedule_function1()

def collect_data():
    global data_points, is_selection
    
    count = 0
    while count < 20:
        data = client.get_next_packet()
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
    is_selection = main_model_testing(data_1)
    data_points = []
    dwellbuffer = hl2ss_rus.command_buffer() 
    if is_selection > 0.5:
        #send_text_to_device('1')
        print('Selection')
        #dwellbuffer = hl2ss_rus.command_buffer() 
        dwellbuffer.connect_success('Selection')
    
        # Send command(s) in buffer to the Unity app and receive response (4 byte unsigned integer per command)
        #ipc.push(dwellbuffer)
        #response = ipc.pull(dwellbuffer)
        
        # Evaluate response
        print("Received response after Selection:")
        #print(response[0])
    else:
        #send_text_to_device('0')
        print('No Selection')
        dwellbuffer.connect_success('No Selection')
        #collect_data()
    
    # Send command(s) in buffer to the Unity app and receive response (4 byte unsigned integer per command)
    ipc.push(dwellbuffer)
    response = ipc.pull(dwellbuffer)
    print(response[0])
    #write_data_to_csv()
    
def write_data_to_csv():
    global data_points
    with open('eye_tracking_data.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the data points to CSV
        writer.writerows(data_points)
    data_points = []  # Clear the list after writing
    
def get_info_from_device():
    global highlight
    if highlight == True:
        start_time = time.time
        collect_data()
        end_time = time.time
        print(end_time-start_time)

def schedule_function1():
    interval = 0.2  # 200 milliseconds
    timer = threading.Timer(interval, inform_connect)
    timer.start()
    return timer

def on_press(keyboard_key):
    global running, timer
    if keyboard_key == keyboard.Key.esc:
        running = False
        if timer:
            timer.cancel()
        print("Checker stopped")
        return False  # Stop the listener

def toggle_value():
    global value_to_check, running
    while running:
        time.sleep(5)  # Change the value every second
        value_to_check = not value_to_check
        print(f"Value changed to {value_to_check}")

stop_event = threading.Event()
ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
ipc.open()

client = hl2ss_lnm.rx_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, fps=fps)
client.open()

# Add the sub_folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'EyeTracking'))

try:
    from ModelTesting import main_model_testing  # Import the function
    #print("Successfully imported process_data from test2.py")
except ImportError as e:
    print(f"Error importing model testing")

# Start the repeated function call
timer = schedule_function1()

# Thread for toggling the value every second
toggle_value_thread = threading.Thread(target=toggle_value)
toggle_value_thread.start()

# Start listening for the escape key press to stop checking
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

# Wait for the toggle_value_thread to finish
toggle_value_thread.join()
    
listener.join()
ipc.close()
client.close()