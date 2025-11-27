import time
import os

# os.system("setxkbmap us")
time.sleep(5)
import pyautogui

"""
while True:
    pyautogui.displayMousePosition()
    time.sleep(0.1)
"""

def photo(biome):
    pyautogui.press('t')
    pyautogui.write(f"/locate biome minecraft:{biome}", interval=0.02)
    pyautogui.press('ENTER')
    time.sleep(5)
    pyautogui.press('t')
    time.sleep(0.5)
    pyautogui.moveTo(1074, 1233)
    pyautogui.click()
    pyautogui.press('ENTER')
    time.sleep(10)
    pyautogui.press('F1')
    time.sleep(1)
    pyautogui.press('F2')
    time.sleep(1)
    pyautogui.press('F1')


biomes=open("biomes.txt","r")
for biome in biomes:
    photo(biome)

# os.system("setxkbmap tr")