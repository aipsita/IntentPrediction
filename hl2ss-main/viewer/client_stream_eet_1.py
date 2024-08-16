import csv
from pynput import keyboard
import hl2ss
import hl2ss_lnm

# Settings --------------------------------------------------------------------

# HoloLens address
host = '192.168.1.179'

# Target Frame Rate
# Options: 30, 60, 90
fps = 90

#------------------------------------------------------------------------------

enable = True
target_state = False  # Initialize the target state

def on_press(key):
    global enable, target_state
    if key == keyboard.Key.esc:
        enable = False
    elif key.char == '1':
        target_state = not target_state  # Toggle target state on space bar press
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

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

    while (enable):
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
