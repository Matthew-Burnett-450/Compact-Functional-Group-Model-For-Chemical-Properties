
import pyautogui
import time
hydrocarbons = [
    "4-methyloctane",
    "2,5-dimethyloctane",
    "3,6-dimethyloctane",
    "2-methylnonane",
    "3-methylnonane",
    "2,5-dimethylnonane",
    "3-methyldecane",
    "2-methyldecane",
    "4-methyldecane",
    "n-undecane",
    "3-methylundecane",
    "5-methylundecane"
]

def perform_macro(names):
    for name in names:
        print(name)
        pyautogui.hotkey('ctrl', 'n')  # Press Ctrl+N to create a new document
        time.sleep(1)  # Wait a bit for the new document to open
        pyautogui.typewrite(name)  # Type the name
        time.sleep(1)
        pyautogui.press('enter')
        time.sleep(1)
        pyautogui.hotkey('ctrl', 'g')  # Press Ctrl+G (usually 'go to' in text editors)

        time.sleep(10)  # Wait for 3 seconds
        pyautogui.press('right')  # Press the right arrow key
        pyautogui.press('enter')
        pyautogui.hotkey('ctrl', 's')  # Press Ctrl+S to save
        time.sleep(1)  # Wait a bit for the save dialog to open
        pyautogui.typewrite(name)  # Type the name
        time.sleep(.5)
        pyautogui.press('enter')  # Press Enter to confirm the save
        time.sleep(3)
time.sleep(3)
perform_macro(hydrocarbons)