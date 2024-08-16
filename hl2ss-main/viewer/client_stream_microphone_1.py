from pynput import keyboard
import hl2ss
import hl2ss_lnm
import hl2ss_utilities
import pyaudio
import queue
import threading
import numpy as np
import scipy.fftpack

# Settings --------------------------------------------------------------------

# HoloLens address
host = "10.25.14.181"

# Audio encoding profile
profile = hl2ss.AudioProfile.AAC_24000

#------------------------------------------------------------------------------

# RAW format is s16 packed, AAC decoded format is f32 planar
audio_format = pyaudio.paInt16 if (profile == hl2ss.AudioProfile.RAW) else pyaudio.paFloat32
enable = True

def pcmworker(pcmqueue):
    global enable
    global audio_format
    p = pyaudio.PyAudio()
    stream = p.open(format=audio_format, channels=hl2ss.Parameters_MICROPHONE.CHANNELS, rate=hl2ss.Parameters_MICROPHONE.SAMPLE_RATE, output=True)
    stream.start_stream()
    while enable:
        stream.write(pcmqueue.get())
    stream.stop_stream()
    stream.close()

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

def is_speaking(audio_data, threshold=0.01):
    """
    Determines if the user is speaking based on the RMS value of the audio signal.
    
    Parameters:
    - audio_data: The audio data in PCM format.
    - threshold: The RMS threshold to determine if speech is present.
    
    Returns:
    - True if speech is detected, False otherwise.
    """
    # Convert byte data to numpy array
    audio_signal = np.frombuffer(audio_data, dtype=np.float32)
    
    # Calculate the RMS value
    rms = np.sqrt(np.mean(audio_signal**2))
    
    # Determine if the RMS value exceeds the threshold
    return rms > threshold

def reduce_noise(audio_signal, spectral_threshold=0.02, volume_threshold=0.01):
    """
    Applies a spectral gating noise reduction filter and volume-based reduction to the audio signal.
    
    Parameters:
    - audio_signal: The audio signal as a numpy array.
    - spectral_threshold: The spectral threshold below which the signal is considered noise and reduced.
    - volume_threshold: The RMS threshold below which the signal is considered noise and reduced.
    
    Returns:
    - The noise-reduced audio signal as a numpy array.
    """
    # Perform FFT
    fft_signal = scipy.fftpack.fft(audio_signal)
    
    # Calculate magnitude and phase
    magnitude = np.abs(fft_signal)
    phase = np.angle(fft_signal)
    
    # Apply spectral threshold
    magnitude[magnitude < spectral_threshold] = 0
    
    # Reconstruct signal
    fft_signal = magnitude * np.exp(1j * phase)
    audio_signal_clean = np.real(scipy.fftpack.ifft(fft_signal))
    
    # Calculate the RMS value of the cleaned audio signal
    rms = np.sqrt(np.mean(audio_signal_clean**2))
    
    # Apply volume threshold
    if rms < volume_threshold:
        audio_signal_clean *= 0.5  # Reduce volume by half if below volume threshold
    
    return audio_signal_clean

pcmqueue = queue.Queue()
thread = threading.Thread(target=pcmworker, args=(pcmqueue,))
listener = keyboard.Listener(on_press=on_press)
thread.start()
listener.start()

client = hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE, profile=profile)
client.open()

while enable: 
    data = client.get_next_packet()
    # RAW format is s16 packed, AAC decoded format is f32 planar
    audio = hl2ss_utilities.microphone_planar_to_packed(data.payload) if (profile != hl2ss.AudioProfile.RAW) else data.payload
    
    # Ensure the audio array is contiguous
    audio_contiguous = np.ascontiguousarray(audio)
    
    # Convert to numpy array for noise reduction
    audio_signal = np.frombuffer(audio_contiguous, dtype=np.float32)
    
    # Apply noise reduction
    audio_clean = reduce_noise(audio_signal)
    
    # Convert back to bytes and put in queue
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
