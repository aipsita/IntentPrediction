import csv
from pynput import keyboard
import threading
import hl2ss
import hl2ss_lnm
import hl2ss_rus

# Settings --------------------------------------------------------------------

# HoloLens address
host = '192.168.1.179'

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

stop_event = threading.Event()
ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
ipc.open()

key = 0  # Initialize key

def on_press(keyboard_key):
    global enable, target_state, key
    if keyboard_key == keyboard.Key.esc:
        enable = False
        stop_event.set()
    elif keyboard_key.char == '1':
        target_state = not target_state  # Toggle target state on key press
        inform_connect()
        #threading.Thread(target=send_text_to_device, args=('1',)).start()  # Run send_text_to_device in a separate thread
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

# Message functions -----------------------------------------------------------
# Inform server about succuessful connect
def inform_connect():
    # Create command buffer and append command(s)
    buffer = hl2ss_rus.command_buffer() 
    buffer.connect_success()
    
    # Send command(s) in buffer to the Unity app and receive response (4 byte unsigned integer per command)
    ipc.push(buffer)
    response = ipc.pull(buffer)
    
    # Evaluate response
    print("Received response:")
    print(response)

def send_text_to_device(text):
    global key
    display_list = hl2ss_rus.command_buffer()
    #display_list.inform_connect() # Connect success
    display_list.begin_display_list() # Begin command sequence
    display_list.remove_all() # Remove all objects that were created remotely
    display_list.create_text() # Create text object, server will return its id
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast) # Set server to use the last created object as target, this avoids waiting for the id of the text object
    display_list.set_text(key, font_size, rgba, text) # Set text
    display_list.set_world_transform(key, position, rotation, [1, 1, 1]) # Set the world transform of the text object
    display_list.set_active(key, hl2ss_rus.ActiveState.Active) # Make the text object visible
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseID) # Restore target mode
    display_list.end_display_list() # End command sequence
    ipc.push(display_list) # Send commands to server
    results = ipc.pull(display_list) # Get results from server
    key = results[2] # Get the text object id, created by the 3rd command in the list
    print(f'Created text object with id {key}')

client = hl2ss_lnm.rx_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, fps=fps)
client.open()

# Open CSV file for writing
with open('eye_tracking_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow([
        'Timestamp', 'Pose', 'Calibration_Valid',
        'Combined_Eye_Gaze_Valid', 'Combined_Eye_Gaze_Origin', 'Combined_Eye_Gaze_Direction',
        'Left_Eye_Gaze_Valid', 'Left_Eye_Gaze_Origin', 'Left_Eye_Gaze_Direction',
        'Right_Eye_Gaze_Valid', 'Right_Eye_Gaze_Origin', 'Right_Eye_Gaze_Direction',
        'Left_Eye_Openness_Valid', 'Left_Eye_Openness',
        'Right_Eye_Openness_Valid', 'Right_Eye_Openness',
        'Vergence_Distance_Valid', 'Vergence_Distance', 'Target'
    ])

    while enable:
        data = client.get_next_packet()
        eet = hl2ss.unpack_eet(data.payload)

        # Write the data to CSV
        writer.writerow([
            data.timestamp, data.pose, eet.calibration_valid,
            eet.combined_ray_valid, eet.combined_ray.origin, eet.combined_ray.direction,
            eet.left_ray_valid, eet.left_ray.origin, eet.left_ray.direction,
            eet.right_ray_valid, eet.right_ray.origin, eet.right_ray.direction,
            eet.left_openness_valid, eet.left_openness,
            eet.right_openness_valid, eet.right_openness,
            eet.vergence_distance_valid, eet.vergence_distance, target_state
        ])

client.close()
listener.join()
ipc.close()