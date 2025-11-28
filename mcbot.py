import time
import os
import pyautogui

time.sleep(5)

# Settings
CHAT_TP_CLICK_POS = (1074, 1233)  # coordinate to click after locate biome command
WAIT_AFTER_LOCATE = 5             # wait for /locate command to run
WAIT_AFTER_TP = 5                 # chunk load time
WAIT_FALL = 5                     # wait time for the fall
WAIT_AFTER_Y_MOVE = 3             # /tp ~ ~100 ~
LOOP_MAX_ITERATION = 500          # 

BASE_DIR = "data"

# Doube jump to enable flying mode  
def press_double_space():
    for _ in range(2):
        pyautogui.keyDown('space')
        time.sleep(0.03)   # tuşa basılı durma süresi
        pyautogui.keyUp('space')
        time.sleep(0.01)   # iki basış arası süre

# Run minecraft commands
def run_command(cmd: str, interval: float = 0.02, wait_after: float = 0.5):
    pyautogui.press('t')
    pyautogui.write(cmd, interval=interval)
    pyautogui.press('enter')
    time.sleep(wait_after)

# Locate biome and teleport
def go_to_biome(biome: str):
    # /locate biome minecraft:<biome>
    run_command(f"/locate biome minecraft:{biome}", wait_after=WAIT_AFTER_LOCATE)

    # Click the coordinates on the chat and teleport
    pyautogui.press('t')
    time.sleep(0.5)
    pyautogui.moveTo(*CHAT_TP_CLICK_POS)
    pyautogui.click()
    pyautogui.press('enter')

    time.sleep(WAIT_AFTER_TP)

# Take position for screenshot, 60 blocks higher than ground height
def position_for_screenshot():
    
    # 1) Get yourself 100 blocks high
    run_command("/tp ~ ~100 ~", wait_after=WAIT_AFTER_Y_MOVE)

    # 2) Double jump to stop flying and fall down
    press_double_space()
    time.sleep(WAIT_FALL)

    # 3) Start flying again
    press_double_space()
    time.sleep(0.5)

    # 4) Get yourself 60 blocks up
    run_command("/tp ~ ~60 ~", wait_after=WAIT_AFTER_Y_MOVE)

# Save screenshot as data/<biome>/<cycle_index>.png
def take_screenshot(biome: str, cycle_index: int):

    biome = biome.strip()
    # Create folders
    biome_dir = os.path.join(BASE_DIR, biome)
    os.makedirs(biome_dir, exist_ok=True)

    # Hid HUD
    pyautogui.press('f1')
    time.sleep(1)

    # Screenshot with pyautogui
    filename = os.path.join(biome_dir, f"{cycle_index}.png")
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)

    # Open HUD back
    time.sleep(1)
    pyautogui.press('f1')

# One biome loop
def process_biome(biome: str, cycle_index: int):

    biome = biome.strip()
    if not biome:
        return

    # Go to biome
    go_to_biome(biome)

    # Take position
    position_for_screenshot()

    # Take screenshot
    take_screenshot(biome, cycle_index)

# Change 30000 blocks
def move_30000_x():
    run_command("/tp ~30000 ~ ~", wait_after=WAIT_AFTER_TP)

def main():
    # Load biome list
    with open("biomes.txt", "r") as f:
        biomes = [line.strip() for line in f if line.strip()]

    cycle_index = 0

    # The loop
    while True:
        for biome in biomes:
            process_biome(biome, cycle_index)

        # Move 30000 blocks to find new biomes
        move_30000_x()

        cycle_index += 1

        if cycle_index > LOOP_MAX_ITERATION:
            break

if __name__ == "__main__":
    time.sleep(5)
    main()