import numpy as np
import pandas as pd
from pynput import keyboard
import hl2ss
import hl2ss_lnm

# Settings --------------------------------------------------------------------

# HoloLens address
host = "10.25.14.181"

# CSV output file
OUTPUT_FILE = 'hand_joint_data.csv'

# Initialize variables
enable = True
columns = ['Timestamp', 'Hand', 'Joint', 'Position_X', 'Position_Y', 'Position_Z', 'Orientation_X', 'Orientation_Y', 'Orientation_Z', 'Orientation_W']
data_df = pd.DataFrame(columns=columns)

# Joint names for clarity
joint_names = [
    "Palm", "Wrist", "ThumbMetacarpal", "ThumbProximal", "ThumbDistal", "ThumbTip",
    "IndexMetacarpal", "IndexProximal", "IndexIntermediate", "IndexDistal", "IndexTip",
    "MiddleMetacarpal", "MiddleProximal", "MiddleIntermediate", "MiddleDistal", "MiddleTip",
    "RingMetacarpal", "RingProximal", "RingIntermediate", "RingDistal", "RingTip",
    "LittleMetacarpal", "LittleProximal", "LittleIntermediate", "LittleDistal", "LittleTip"
]

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

client = hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT)
client.open()

# Define an empty DataFrame
data_df = pd.DataFrame(columns=columns)

# Use concat to add a new row
def append_to_dataframe(timestamp, hand, joint, position, orientation):
    row = pd.DataFrame([{
        'Timestamp': timestamp,
        'Hand': hand,
        'Joint': joint_names[joint],
        'Position_X': position[0],
        'Position_Y': position[1],
        'Position_Z': position[2],
        'Orientation_X': orientation[0],
        'Orientation_Y': orientation[1],
        'Orientation_Z': orientation[2],
        'Orientation_W': orientation[3]
    }])
    global data_df
    data_df = pd.concat([data_df, row], ignore_index=True)

while enable:
    data = client.get_next_packet()
    si = hl2ss.unpack_si(data.payload)

    if si.is_valid_hand_left():
        hand_left = si.get_hand_left()
        for joint in range(hl2ss.SI_HandJointKind.TOTAL):
            pose = hand_left.get_joint_pose(joint)
            append_to_dataframe(data.timestamp, 'Left', joint, pose.position, pose.orientation)

    if si.is_valid_hand_right():
        hand_right = si.get_hand_right()
        for joint in range(hl2ss.SI_HandJointKind.TOTAL):
            pose = hand_right.get_joint_pose(joint)
            append_to_dataframe(data.timestamp, 'Right', joint, pose.position, pose.orientation)

client.close()
listener.join()

# Save data to CSV
data_df.to_csv(OUTPUT_FILE, index=False)
