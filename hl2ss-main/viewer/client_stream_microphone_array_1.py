import numpy as np
import hl2ss
import hl2ss_lnm
import pyaudio
import queue
import threading
import wave
from pynput import keyboard
from vosk import Model, KaldiRecognizer

# Settings --------------------------------------------------------------------

# HoloLens address
host = "10.25.14.181"

# Channel
channel = hl2ss.Parameters_MICROPHONE.ARRAY_TOP_LEFT

# Audio format
audio_format = pyaudio.paFloat32
enable = True

# Function to process audio and perform noise reduction
def reduce_noise(audio_signal):
    # Simple noise reduction logic (e.g., spectral gating, band-pass filter)
    # This is a placeholder and should be replaced with actual noise reduction logic
    return audio_signal

def pcmworker(pcmqueue):
    global enable
    global audio_format
    p = pyaudio.PyAudio()
    stream = p.open(format=audio_format, channels=1, rate=hl2ss.Parameters_MICROPHONE.SAMPLE_RATE, output=True)
    stream.start_stream()
    while enable:
        stream.write(pcmqueue.get())
    stream.stop_stream()
    stream.close()

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

def is_speaking(audio_data, threshold=20):
    audio_signal = np.frombuffer(audio_data, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_signal**2))
    return rms > threshold

pcmqueue = queue.Queue()
thread = threading.Thread(target=pcmworker, args=(pcmqueue,))
listener = keyboard.Listener(on_press=on_press)
thread.start()
listener.start()

client = hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE, profile=hl2ss.AudioProfile.RAW, level=hl2ss.AACLevel.L5)
client.open()

while enable: 
    data = client.get_next_packet()
    audio = data.payload[0, channel::hl2ss.Parameters_MICROPHONE.ARRAY_CHANNELS]
    
    # Ensure the audio array is contiguous
    audio_contiguous = np.ascontiguousarray(audio)
    
    # Apply noise reduction
    audio_clean = reduce_noise(audio_contiguous)
    
    # Convert to bytes and put in queue
    pcmqueue.put(audio_clean.tobytes())
    
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
