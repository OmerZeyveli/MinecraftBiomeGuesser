import time
import os
import pyautogui



# ================= SETTINGS =================

CHAT_TP_CLICK_POS = (745, 870)
WAIT_AFTER_LOCATE = 3
WAIT_AFTER_TP = 5
WAIT_FALL = 5
WAIT_AFTER_Y_MOVE = 3
LOOP_MAX_ITERATION = 500
SS_HEIGHT = 10

BASE_DIR = "data"
LOG_FILE = os.path.expanduser("/home/unre/.minecraft/logs/latest.log")

# ============================================

def press_double_space():
    for _ in range(2):
        pyautogui.keyDown('space')
        time.sleep(0.03)
        pyautogui.keyUp('space')
        time.sleep(0.01)

def run_command(cmd: str, interval: float = 0.005, wait_after: float = 0.5):
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
        wait_after=0.1
    )
    return check_log_for(keyword)

# ================= BIOME CENTER =================

def center_in_biome(biome: str, step: int = 20):
    temp = []
    for i in range(1,5):
        north = biome_check_direction(biome, 0, -step, "N")
        south = biome_check_direction(biome, 0, step, "S")
        east  = biome_check_direction(biome, step, 0, "E")
        west  = biome_check_direction(biome, -step, 0, "W")

        dx = dz = 0
        at_least_one_true = False
        at_center = True
        if east:
            dx += step
            at_least_one_true = True
        else:
            at_center = False
        if west:
            dx -= step
            at_least_one_true = True
        else:
            at_center = False
        if south:
            dz += step
            at_least_one_true = True
        else:
            at_center = False
        if north:
            dz -= step
            at_least_one_true = True
        else:
            at_center = False
        if not at_least_one_true:
            return False
        if at_center:
            return True
        command = f"/tp ~{dx} ~ ~{dz}"
        if command in temp:
            return False
        temp.append(command)
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
