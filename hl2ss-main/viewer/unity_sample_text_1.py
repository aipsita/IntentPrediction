from pynput import keyboard
import threading
import hl2ss
import hl2ss_lnm
import hl2ss_rus

# Settings --------------------------------------------------------------------

# HoloLens address
host = '10.25.10.186'

# Position in world space (x, y, z) in meters
position = [0, 0, 1]

# Rotation in world space (x, y, z, w) as a quaternion
rotation = [0, 0, 0, 1]

# Font size
font_size = 0.4

# Text color
rgba = [1, 1, 1, 1]

#------------------------------------------------------------------------------

stop_event = threading.Event()

# Connection to Unity
ipc = hl2ss_lnm.ipc_umq(host, hl2ss.IPCPort.UNITY_MESSAGE_QUEUE)
ipc.open()

def on_press(key):
    try:
        if key == keyboard.Key.esc:
            stop_event.set()
            return False
        elif key.char == '1':
            send_text_to_unity('1')
    except AttributeError:
        pass  # Handle special keys that do not have .char attribute

def send_text_to_unity(text):
    display_list = hl2ss_rus.command_buffer()
    display_list.begin_display_list()
    display_list.remove_all()
    display_list.create_text()
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseLast)
    display_list.set_text(0, font_size, rgba, text)
    display_list.set_world_transform(0, position, rotation, [1, 1, 1])
    display_list.set_active(0, hl2ss_rus.ActiveState.Active)
    display_list.set_target_mode(hl2ss_rus.TargetMode.UseID)
    display_list.end_display_list()
    ipc.push(display_list)
    results = ipc.pull(display_list)
    key = results[2]
    print(f'Created text object with id {key}')

key = 0

listener = keyboard.Listener(on_press=on_press)
listener.start()

stop_event.wait()

command_buffer = hl2ss_rus.command_buffer()
command_buffer.remove(key)  # Destroy text object
ipc.push(command_buffer)
results = ipc.pull(command_buffer)

ipc.close()

listener.join()
