import time
import os
import pyautogui



# ================= SETTINGS =================

CHAT_TP_CLICK_POS = (745, 870)
WAIT_AFTER_LOCATE = 7
WAIT_AFTER_TP = 5
WAIT_FALL = 8
WAIT_AFTER_Y_MOVE = 2
LOOP_MAX_ITERATION = 500000
SS_HEIGHT = 15

BASE_DIR = "data"
LOG_FILE = os.path.expanduser("/home/unre/.minecraft/logs/latest.log")

# ============================================

def press_double_space():
    for _ in range(2):
        pyautogui.keyDown('space')
        time.sleep(0.03)
        pyautogui.keyUp('space')
        time.sleep(0.01)

def run_command(cmd: str, interval: float = 0.001, wait_after: float = 0.5):
    pyautogui.press('t')
    pyautogui.write(cmd, interval=interval)
    pyautogui.press('enter')
    time.sleep(wait_after)

# ================= LOG READ =================

   
def check_log_for(keyword: str):
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            last_line = f.readlines()[-1].rstrip("\n")
            return keyword in last_line
    except (FileNotFoundError, IndexError):
        return False

def biome_check_direction(biome: str, dx: int, dz: int, tag: str):
    keyword = f"BIOME_{tag}_TRUE"
    run_command(
        f"/execute if biome ~{dx} ~ ~{dz} minecraft:{biome} run say {keyword}",
        wait_after=0.15
    )
    return check_log_for(keyword)

# ================= BIOME CENTER =================

def center_in_biome(biome: str, step: int = 24):
    went_to_east = False
    went_to_west = False
    went_to_north = False
    went_to_south = False
    at_center = False
    for i in range(1,8):
        north = biome_check_direction(biome, 0, -step, "N")
        south = biome_check_direction(biome, 0, step, "S")
        east  = biome_check_direction(biome, step, 0, "E")
        west  = biome_check_direction(biome, -step, 0, "W")

        dx = dz = 0
        at_least_one_true = False
        
        if east and not went_to_west:
            dx += step
            at_least_one_true = True
            went_to_east = True

        if west and not went_to_east:
            dx -= step
            at_least_one_true = True
            went_to_west = True

        if south and not went_to_north:
            dz += step
            at_least_one_true = True
            went_to_south = True

        if north and not went_to_south:
            dz -= step
            at_least_one_true = True
            went_to_north = True

        if not at_least_one_true:
            return False
        at_center = east and west and south and north
        if at_center:
            return True
        command = f"/tp ~{dx} ~ ~{dz}"
        run_command(command, wait_after=WAIT_AFTER_TP)

# ================= CORE LOGIC =================

def go_to_biome(biome: str):
    run_command(f"/locate biome minecraft:{biome}", wait_after=WAIT_AFTER_LOCATE)

    pyautogui.press('t')
    time.sleep(0.5)
    pyautogui.moveTo(*CHAT_TP_CLICK_POS)
    pyautogui.click()
    pyautogui.press('enter')

    time.sleep(WAIT_AFTER_TP)

def position_for_screenshot():
    run_command("/tp ~ ~100 ~", wait_after=WAIT_AFTER_Y_MOVE)

    press_double_space()
    time.sleep(WAIT_FALL)

    press_double_space()
    time.sleep(0.5)

    run_command(f"/tp ~ ~{SS_HEIGHT} ~", wait_after=WAIT_AFTER_Y_MOVE)

def take_screenshot(biome: str, cycle_index: int):
    biome = biome.strip()
    biome_dir = os.path.join(BASE_DIR, biome)
    os.makedirs(biome_dir, exist_ok=True)

    pyautogui.press('f1')
    time.sleep(1)

    filename = os.path.join(biome_dir, f"{cycle_index}.png")
    pyautogui.screenshot().save(filename)

    time.sleep(1)
    pyautogui.press('f1')

def process_biome(biome: str, cycle_index: int):
    biome = biome.strip()
    if not biome:
        return

    go_to_biome(biome)

    # ⭐ BIOME BORDER FIX

    if not center_in_biome(biome):
        return

    position_for_screenshot()
    take_screenshot(biome, cycle_index)

def move_30000_x():
    run_command("/tp ~30000 ~ ~", wait_after=WAIT_AFTER_TP)

# ================= MAIN =================

def main():
    with open("biomes.txt", "r") as f:
        biomes = [line.strip() for line in f if line.strip()]

    cycle_index = 0

    while True:
        for biome in biomes:
            process_biome(biome, cycle_index)

        move_30000_x()
        cycle_index += 1

        if cycle_index > LOOP_MAX_ITERATION:
            break

if __name__ == "__main__":
    time.sleep(3)
    main()
