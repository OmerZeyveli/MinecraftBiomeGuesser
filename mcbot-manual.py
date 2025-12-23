import time
import os
import glob
import random
import re
from datetime import datetime
import pyautogui

# Hotkey listener: prefer pynput; fallback keyboard
HOTKEY_BACKEND = "pynput"
try:
    from pynput import keyboard as pynput_keyboard
except Exception:
    HOTKEY_BACKEND = "keyboard"
    import keyboard as kb  # windows'ta admin isteyebilir

# =========================
# SETTINGS
# =========================
CHAT_TP_CLICK_POS = (1000, 1225)   # locate çıktısındaki "Teleport" link pozisyonu
BASE_DIR = "data"

WAIT_AFTER_LOCATE = 6.0            # locate sonucunun chat'e gelmesi için
WAIT_AFTER_TP = 3.5                # tp sonrası chunk yükleme

# Locate fail olursa uzak sıçra
MAX_LOCATE_RETRIES = 3
FAR_JUMP_MIN = 80000
FAR_JUMP_MAX = 160000

STARTUP_DELAY = 3.0

# =========================
# LOG PATH (only to detect locate fail)
# =========================
def _latest_log_candidates():
    envp = os.environ.get("MINECRAFT_LOG_PATH")
    if envp:
        return [envp]

    cands = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        cands.append(os.path.join(appdata, ".minecraft", "logs", "latest.log"))
        prism = os.path.join(appdata, "PrismLauncher", "instances", "*", "minecraft", "logs", "latest.log")
        cands.extend(glob.glob(prism))

    home = os.path.expanduser("~")
    cands.append(os.path.join(home, ".minecraft", "logs", "latest.log"))
    cands.append(os.path.join(home, "Library", "Application Support", "minecraft", "logs", "latest.log"))
    return cands

def find_latest_log_path() -> str:
    cands = [p for p in _latest_log_candidates() if p and os.path.exists(p)]
    if not cands:
        return ""
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

LATEST_LOG_PATH = find_latest_log_path()

# Regex: locate OK / FAIL
LOCATE_OK_PAT = re.compile(r"The nearest minecraft:", re.IGNORECASE)
LOCATE_FAIL_PAT = re.compile(r"couldn't find a biome of type", re.IGNORECASE)

def log_mark() -> int:
    if not LATEST_LOG_PATH:
        return -1
    try:
        with open(LATEST_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(0, os.SEEK_END)
            return f.tell()
    except Exception:
        return -1

def log_wait_locate_result(start_pos: int, timeout: float = 10.0) -> str:
    """
    Returns: "OK" / "FAIL" / "UNKNOWN"
    If log not available, returns "UNKNOWN".
    """
    if not LATEST_LOG_PATH or start_pos < 0:
        return "UNKNOWN"

    start_t = time.time()
    with open(LATEST_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(start_pos)
        while time.time() - start_t < timeout:
            line = f.readline()
            if not line:
                time.sleep(0.05)
                continue
            if LOCATE_FAIL_PAT.search(line):
                return "FAIL"
            if LOCATE_OK_PAT.search(line):
                return "OK"
    return "UNKNOWN"

# =========================
# INPUT HELPERS
# =========================
def run_command(cmd: str, interval: float = 0.01, wait_after: float = 0.15):
    pyautogui.press('t')
    pyautogui.write(cmd, interval=interval)
    pyautogui.press('enter')
    time.sleep(wait_after)

def far_jump():
    dx = random.choice([-1, 1]) * random.randint(FAR_JUMP_MIN, FAR_JUMP_MAX)
    dz = random.choice([-1, 1]) * random.randint(FAR_JUMP_MIN, FAR_JUMP_MAX)
    run_command(f"/tp ~{dx} ~ ~{dz}", wait_after=0.2)
    time.sleep(WAIT_AFTER_TP)

# =========================
# SCREENSHOT
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def next_filename(biome: str) -> str:
    biome_dir = os.path.join(BASE_DIR, biome)
    ensure_dir(biome_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(biome_dir, f"{stamp}.png")

def take_screenshot(biome: str):
    fn = next_filename(biome)
    pyautogui.screenshot().save(fn)
    print(f"[SHOT] {fn}")

# =========================
# LOCATE + TP (manual)
# =========================
def locate_and_tp(biome: str) -> bool:
    """
    /locate biome minecraft:<biome> -> click tp link -> tp
    If fail (from log), far jump and retry.
    If log unavailable, just attempts without fail detection.
    """
    for attempt in range(1, MAX_LOCATE_RETRIES + 1):
        start_pos = log_mark()
        run_command(f"/locate biome minecraft:{biome}", wait_after=0.2)
        time.sleep(WAIT_AFTER_LOCATE)

        res = log_wait_locate_result(start_pos, timeout=10.0)
        if res == "FAIL":
            print(f"[LOCATE] {biome}: FAIL (attempt {attempt}/{MAX_LOCATE_RETRIES}) -> far jump")
            far_jump()
            continue

        # OK veya UNKNOWN: tp linkine tıkla (UNKNOWN durumda da deniyoruz)
        pyautogui.press('t')
        time.sleep(0.25)
        pyautogui.moveTo(*CHAT_TP_CLICK_POS)
        pyautogui.click()
        pyautogui.press('enter')
        time.sleep(WAIT_AFTER_TP)
        return True

    print(f"[SKIP] {biome}: locate failed too many times.")
    return False

# =========================
# MANUAL CONTROLLER
# =========================
class ManualController:
    def __init__(self, biomes: list[str]):
        self.biomes = biomes
        self.idx = 0
        self.current = self.biomes[self.idx]
        self.busy = False

    def show_help(self):
        print("\n=== MANUAL MODE (NO SCOREBOARD) ===")
        print(f"Biome: {self.current} ({self.idx+1}/{len(self.biomes)})")
        print("F8  -> Screenshot (data/<biome>/)")
        print("F9  -> Next biome (locate+tp)")
        print("F7  -> Prev biome (locate+tp)")
        print("f12 -> Quit")
        print("==================================\n")

    def goto_current(self):
        if self.busy:
            return
        self.busy = True
        try:
            print(f"[GO] {self.current}")
            ok = locate_and_tp(self.current)
            if not ok:
                print(f"[WARN] {self.current} skipped.")
        finally:
            self.busy = False
            self.show_help()

    def shot(self):
        if self.busy:
            return
        take_screenshot(self.current)

    def next_biome(self):
        if self.busy:
            return
        self.idx = (self.idx + 1) % len(self.biomes)
        self.current = self.biomes[self.idx]
        self.goto_current()

    def prev_biome(self):
        if self.busy:
            return
        self.idx = (self.idx - 1) % len(self.biomes)
        self.current = self.biomes[self.idx]
        self.goto_current()

def load_biomes(path="biomes.txt") -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        biomes = [line.strip() for line in f if line.strip()]
    if not biomes:
        raise ValueError("biomes.txt boş.")
    return biomes

# =========================
# HOTKEYS
# =========================
def main():
    time.sleep(STARTUP_DELAY)
    biomes = load_biomes()
    ctl = ManualController(biomes)

    ctl.show_help()
    ctl.goto_current()

    if HOTKEY_BACKEND == "pynput":
        print("[INFO] Hotkeys backend: pynput")

        def on_press(key):
            try:
                if key == pynput_keyboard.Key.f8:
                    ctl.shot()
                elif key == pynput_keyboard.Key.f9:
                    ctl.next_biome()
                elif key == pynput_keyboard.Key.f7:
                    ctl.prev_biome()
                elif key == pynput_keyboard.Key.f12:
                    print("[EXIT] bye")
                    return False
            except Exception as e:
                print("[ERROR]", e)

        with pynput_keyboard.Listener(on_press=on_press) as listener:
            listener.join()
    else:
        print("[INFO] Hotkeys backend: keyboard")
        kb.add_hotkey("f8", ctl.shot)
        kb.add_hotkey("f9", ctl.next_biome)
        kb.add_hotkey("f7", ctl.prev_biome)
        print("Press f12 to quit.")
        kb.wait("f12")

if __name__ == "__main__":
    main()
