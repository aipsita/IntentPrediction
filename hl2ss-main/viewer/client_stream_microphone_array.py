#------------------------------------------------------------------------------
# This script receives RAW microphone audio from the HoloLens microphone array
# and plays it.
# Audio stream configuration is fixed to 5 channels, 48000 Hz, float32.
# Only one channel is played at a time.
# Press esc to stop.
#------------------------------------------------------------------------------

from pynput import keyboard

import numpy as np
import hl2ss
import hl2ss_lnm
import pyaudio
import queue
import threading

# Settings --------------------------------------------------------------------

# HoloLens address
host = "10.25.14.181"

# Channel
# Options:
# hl2ss.Parameters_MICROPHONE.ARRAY_TOP_LEFT
# hl2ss.Parameters_MICROPHONE.ARRAY_TOP_CENTER
# hl2ss.Parameters_MICROPHONE.ARRAY_TOP_RIGHT
# hl2ss.Parameters_MICROPHONE.ARRAY_BOTTOM_LEFT
# hl2ss.Parameters_MICROPHONE.ARRAY_BOTTOM_RIGHT
channel = hl2ss.Parameters_MICROPHONE.ARRAY_TOP_LEFT

#------------------------------------------------------------------------------

audio_format = pyaudio.paFloat32
enable = True

def pcmworker(pcmqueue):
    global enable
    global audio_format
    p = pyaudio.PyAudio()
    stream = p.open(format=audio_format, channels=1, rate=hl2ss.Parameters_MICROPHONE.SAMPLE_RATE, output=True)
    stream.start_stream()
    while (enable):
        stream.write(pcmqueue.get())
    stream.stop_stream()
    stream.close()

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

def is_speaking(audio_data, threshold=20):
    """
    Determines if the user is speaking based on the RMS value of the audio signal.
    
    Parameters:
    - audio_data: The audio data in PCM format.
    - threshold: The RMS threshold to determine if speech is present.
    
    Returns:
    - True if speech is detected, False otherwise.
    """
    # Convert byte data to numpy array
    audio_signal = np.frombuffer(audio_data, dtype=np.int16)
    
    # Calculate the RMS value
    rms = np.sqrt(np.mean(audio_signal**2))
    #print(rms)
    
    # Determine if the RMS value exceeds the threshold
    return rms > threshold

pcmqueue = queue.Queue()
thread = threading.Thread(target=pcmworker, args=(pcmqueue,))
listener = keyboard.Listener(on_press=on_press)
thread.start()
listener.start()

client = hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE, profile=hl2ss.AudioProfile.RAW, level=hl2ss.AACLevel.L5)
client.open()

while (enable): 
    data = client.get_next_packet()
    audio = data.payload[0, channel::hl2ss.Parameters_MICROPHONE.ARRAY_CHANNELS]
    #pcmqueue.put(audio.tobytes())
    
    # Ensure the audio array is contiguous
    audio_contiguous = np.ascontiguousarray(audio)
    
    # Convert to bytes and put in queue
    pcmqueue.put(audio_contiguous.tobytes())
    
    # Get audio data from queue
    audio_data = pcmqueue.get()
    
    # Check if the user is speaking
    if is_speaking(audio_data):
        print("User is speaking")
    else:
        print(" ")

client.close()

enable = False
pcmqueue.put(b'')
thread.join()
listener.join()
