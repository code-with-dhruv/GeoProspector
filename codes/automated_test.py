
import pyautogui, cv2
import numpy as np
from tensorflow.keras.models import load_model
from pynput import keyboard

model = load_model("model.h5")

def capture_screenshot_from_mouse():
    print("Inside")
    mpos = pyautogui.position()
    width, height = 600, 600
    pyautogui.screenshot("screenshot.png", [
        mpos[0] - width, mpos[1] - height, width, height
    ])
    x = np.array([cv2.resize(cv2.imread("screenshot.png"), (225, 225))], dtype=np.uint8)
    y = model.predict(x)[0].to_list()
    print(x, y)
    class_mapping = {'BareLand': 0, 'Commercial': 1, 'DenseResidential': 2, 'Desert': 3, 'Farmland': 4, 'Forest': 5, 'Industrial': 6, 'Meadow': 7, 'MediumResidential': 8, 'Mountain': 9, 'River': 10, 'SparseResidential': 11}
    print("\n", [mpos[0] - width, mpos[1] - height, width, height], "-", class_mapping[y.index(max(y))], "\n")

def on_press(key):
    try:
        if key.char == 's':
            capture_screenshot_from_mouse()
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
print("Starting...")
listener.start()
listener.join()