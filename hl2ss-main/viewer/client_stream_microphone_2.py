from pynput import keyboard
import hl2ss
import hl2ss_lnm
import hl2ss_rus
import hl2ss_utilities
import pyaudio
import queue
import threading
import numpy as np
import scipy.fftpack

# Settings --------------------------------------------------------------------

# HoloLens address
host = "10.25.14.181"
profile = hl2ss.AudioProfile.AAC_24000

# Audio format settings
audio_format = pyaudio.paInt16 if (profile == hl2ss.AudioProfile.RAW) else pyaudio.paFloat32
enable = True

# Initialize Unity IPC
ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
ipc.open()

def pcmworker(pcmqueue):
    global enable, audio_format
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

pcmqueue = queue.Queue()
thread = threading.Thread(target=pcmworker, args=(pcmqueue,))
listener = keyboard.Listener(on_press=on_press)
thread.start()
listener.start()

client = hl2ss_lnm.rx_microphone(host, hl2ss.StreamPort.MICROPHONE, profile=profile)
client.open()

while enable: 
    data = client.get_next_packet()
    audio = hl2ss_utilities.microphone_planar_to_packed(data.payload) if (profile != hl2ss.AudioProfile.RAW) else data.payload
    
    audio_contiguous = np.ascontiguousarray(audio)
    audio_signal = np.frombuffer(audio_contiguous, dtype=np.float32)
    audio_clean = reduce_noise(audio_signal)
    
    pcmqueue.put(audio_clean.tobytes())
    audio_data = pcmqueue.get()
    print("Here")
    
    if is_speaking(audio_data):
        print("User is speaking")
        send_signal_to_unity('UserSpeaking')
    else:
        send_signal_to_unity('NotSpeaking')

client.close()

enable = False
pcmqueue.put(b'')
thread.join()
listener.join()
ipc.close()
