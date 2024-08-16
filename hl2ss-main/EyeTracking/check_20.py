import threading
import time
from pynput import keyboard  # You might need to install this package with `pip install pynput`

# This value will be checked by function1
value_to_check = False
running = True
timer = None

def function2():
    # Your code to be executed when the value is True
    print("Function 2 is called")

def function1():
    global value_to_check, timer

    # Check the value
    if value_to_check:
        # If the value is True, run function2
        function2()

    # Schedule the next check if still running
    if running:
        timer = schedule_function1()

def schedule_function1():
    interval = 0.2  # 200 milliseconds
    timer = threading.Timer(interval, function1)
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
        time.sleep(1)  # Change the value every second
        value_to_check = not value_to_check
        print(f"Value changed to {value_to_check}")

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
