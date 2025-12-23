import time
import os
import glob
import random
import re
import pyautogui

# =========================
# USER SETTINGS
# =========================
PLAYER_NAME = "RiiveYT"          # <-- KENDİ OYUNCU ADIN

CHAT_TP_CLICK_POS = (1000, 1225) # <-- /locate çıktısındaki tp link koordinatı
BASE_DIR = "data"

# Screenshot / uçuş
SS_HEIGHT = 10
WAIT_FALL = 5.0
WAIT_AFTER_Y_MOVE = 2.0

# Locate & teleport
WAIT_AFTER_TP = 4.0              # teleport sonrası chunk yükleme
TP_SETTLE_WAIT = 1.0             # tp sonrası biome check öncesi kısa bekleme

# "Çok resim" modu: az zorlama
R_CHECK = 16                     # kontrol yarıçapı (8 nokta)
MOVE_STEP = 8                    # hareket adımı (8 önerilir)
MAX_STEPS = 3                    # biome başına en fazla kaç yönlü hamle
GLOBAL_MAX_MOVES = 6             # biome başına toplam hamle limiti
DIR_ZERO_FALLBACK_TRIES = 1      # yön belirsizse 1 küçük nudge dene

# Log okuma
TOKEN_TIMEOUT = 25.0             # sende bazen log geç geliyor, düşük yapma
LOG_POLL_SLEEP = 0.05

# Locate retry + uzak sıçra
MAX_LOCATE_RETRIES = 3
LOCATE_RESULT_TIMEOUT = 10.0     # locate sonucunu logdan bekleme
FAR_JUMP_MIN = 60000             # uzak sıçra arttı
FAR_JUMP_MAX = 120000

# Nudge
NUDGE_STEP = 8
NUDGE_RANGE = 32

# Döngü
LOOP_MAX_ITERATION = 500

time.sleep(5)

# =========================
# LOG PATH DETECTION
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
        raise FileNotFoundError("latest.log bulunamadı. MINECRAFT_LOG_PATH env var ile yolu ver.")
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

LATEST_LOG_PATH = find_latest_log_path()
print("[INFO] Using latest.log:", LATEST_LOG_PATH)

# =========================
# LOG TAILER
# =========================
class LogTail:
    def __init__(self, path: str):
        self.path = path
        self.f = open(self.path, "r", encoding="utf-8", errors="ignore")
        self.f.seek(0, os.SEEK_END)

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def mark(self) -> int:
        return self.f.tell()

    def wait_for_token(self, token: str, start_pos: int, timeout: float) -> str:
        self.f.seek(start_pos)
        start = time.time()
        while time.time() - start < timeout:
            line = self.f.readline()
            if not line:
                time.sleep(LOG_POLL_SLEEP)
                continue
            if token in line:
                return line
        raise TimeoutError(f"Token logda yakalanamadı: {token}")

    def wait_for_any_pattern(self, patterns, start_pos: int, timeout: float) -> tuple[str, str] | tuple[None, None]:
        """
        patterns: list[(name, compiled_regex)]
        return: (name, line) or (None, None)
        """
        self.f.seek(start_pos)
        start = time.time()
        while time.time() - start < timeout:
            line = self.f.readline()
            if not line:
                time.sleep(LOG_POLL_SLEEP)
                continue
            for name, pat in patterns:
                if pat.search(line):
                    return (name, line)
        return (None, None)

LOG = LogTail(LATEST_LOG_PATH)

# =========================
# INPUT HELPERS
# =========================
def press_double_space():
    for _ in range(2):
        pyautogui.keyDown('space')
        time.sleep(0.03)
        pyautogui.keyUp('space')
        time.sleep(0.01)

def run_command(cmd: str, interval: float = 0.02, wait_after: float = 0.15):
    pyautogui.press('t')
    pyautogui.write(cmd, interval=interval)
    pyautogui.press('enter')
    time.sleep(wait_after)

# =========================
# SCOREBOARDS
# =========================
def ensure_scoreboards_once():
    run_command("/scoreboard objectives add valid_position dummy", wait_after=0.05)
    run_command("/scoreboard objectives add dirX dummy", wait_after=0.05)
    run_command("/scoreboard objectives add dirZ dummy", wait_after=0.05)

# =========================
# MOVEMENT HELPERS
# =========================
def far_jump():
    dx = random.choice([-1, 1]) * random.randint(FAR_JUMP_MIN, FAR_JUMP_MAX)
    dz = random.choice([-1, 1]) * random.randint(FAR_JUMP_MIN, FAR_JUMP_MAX)
    run_command(f"/tp ~{dx} ~ ~{dz}", wait_after=0.2)
    time.sleep(WAIT_AFTER_TP)

def move_30000_x():
    # tur bitince (her biome turu sonunda) ekstra çeşitlilik için
    run_command("/tp ~30000 ~ ~", wait_after=0.2)
    time.sleep(WAIT_AFTER_TP)

def nudge_random_small():
    dx = random.choice([-1, 1]) * random.randint(NUDGE_STEP, NUDGE_RANGE)
    dz = random.choice([-1, 1]) * random.randint(NUDGE_STEP, NUDGE_RANGE)
    run_command(f"/tp ~{dx} ~ ~{dz}", wait_after=0.1)
    time.sleep(TP_SETTLE_WAIT)

# =========================
# LOCATE + TP
# =========================
LOCATE_OK_PAT = re.compile(r"The nearest minecraft:", re.IGNORECASE)
LOCATE_FAIL_PAT = re.compile(r"couldn't find a biome of type", re.IGNORECASE)

def go_to_biome(biome: str) -> bool:
    """
    locate success -> click tp link -> tp
    locate fail -> far jump + retry
    """
    for attempt in range(1, MAX_LOCATE_RETRIES + 1):
        start_pos = LOG.mark()
        run_command(f"/locate biome minecraft:{biome}", wait_after=0.2)

        which, _line = LOG.wait_for_any_pattern(
            patterns=[("OK", LOCATE_OK_PAT), ("FAIL", LOCATE_FAIL_PAT)],
            start_pos=start_pos,
            timeout=LOCATE_RESULT_TIMEOUT
        )

        if which == "OK":
            # click teleport link in chat
            time.sleep(0.3)
            pyautogui.press('t')
            time.sleep(0.35)
            pyautogui.moveTo(*CHAT_TP_CLICK_POS)
            pyautogui.click()
            pyautogui.press('enter')
            time.sleep(WAIT_AFTER_TP)
            return True

        # FAIL veya TIMEOUT
        print(f"[LOCATE] {biome}: {which or 'TIMEOUT'} (attempt {attempt}/{MAX_LOCATE_RETRIES}) -> far jump")
        far_jump()

    print(f"[SKIP] {biome}: locate başarısız (retries bitti).")
    return False

# =========================
# SAFE CHECK (8 samples) - PLAYER_NAME ile
# =========================
def _send_safe_check(biome: str, R: int, token: str):
    P = PLAYER_NAME

    run_command(f"/scoreboard players set {P} valid_position 1", wait_after=0.02)
    run_command(f"/execute unless biome ~ ~ ~ minecraft:{biome} run scoreboard players set {P} valid_position 0", wait_after=0.02)

    samples = [
        ( R,  0), (-R,  0), ( 0,  R), ( 0, -R),
        ( R,  R), ( R, -R), (-R,  R), (-R, -R),
    ]
    for dx, dz in samples:
        run_command(
            f"/execute if score {P} valid_position matches 1 unless biome ~{dx} ~ ~{dz} minecraft:{biome} "
            f"run scoreboard players set {P} valid_position 0",
            wait_after=0.01
        )

    run_command(f"/execute if score {P} valid_position matches 1 run say VALID {token}", wait_after=0.01)
    run_command(f"/execute if score {P} valid_position matches 0 run say INVALID {token}", wait_after=0.01)

def is_safe_in_biome(biome: str, R: int) -> bool:
    time.sleep(TP_SETTLE_WAIT)
    token = f"SAFE::{biome}::{R}::{time.time_ns()}"
    start_pos = LOG.mark()

    _send_safe_check(biome, R, token)
    line = LOG.wait_for_token(token, start_pos=start_pos, timeout=TOKEN_TIMEOUT)

    # "INVALID" içinde "VALID" geçer -> kelime bazlı kontrol
    if re.search(r"\bINVALID\b", line):
        return False
    if re.search(r"\bVALID\b", line):
        return True
    return False

# =========================
# DIRECTION VOTE (8 samples)
# =========================
def _send_direction_vote(biome: str, R: int, token: str):
    P = PLAYER_NAME
    run_command(f"/scoreboard players set {P} dirX 0", wait_after=0.01)
    run_command(f"/scoreboard players set {P} dirZ 0", wait_after=0.01)

    def apply(dx: int, dz: int, ax: int, az: int):
        # dirX
        if ax > 0:
            run_command(f"/execute if biome ~{dx} ~ ~{dz} minecraft:{biome} run scoreboard players add {P} dirX {ax}", wait_after=0.005)
        elif ax < 0:
            run_command(f"/execute if biome ~{dx} ~ ~{dz} minecraft:{biome} run scoreboard players remove {P} dirX {abs(ax)}", wait_after=0.005)

        # dirZ
        if az > 0:
            run_command(f"/execute if biome ~{dx} ~ ~{dz} minecraft:{biome} run scoreboard players add {P} dirZ {az}", wait_after=0.005)
        elif az < 0:
            run_command(f"/execute if biome ~{dx} ~ ~{dz} minecraft:{biome} run scoreboard players remove {P} dirZ {abs(az)}", wait_after=0.005)

    # Cardinal
    apply( R,  0, +1,  0)  # E
    apply(-R,  0, -1,  0)  # W
    apply( 0,  R,  0, +1)  # S
    apply( 0, -R,  0, -1)  # N

    # Diagonals
    apply( R, -R, +1, -1)  # NE
    apply(-R, -R, -1, -1)  # NW
    apply( R,  R, +1, +1)  # SE
    apply(-R,  R, -1, +1)  # SW

    run_command(f"/scoreboard players get {P} dirX", wait_after=0.02)
    run_command(f"/scoreboard players get {P} dirZ", wait_after=0.02)
    run_command(f"/say DIRTOKEN {token}", wait_after=0.01)

def compute_direction(biome: str, R: int) -> tuple[int, int]:
    time.sleep(TP_SETTLE_WAIT)
    token = f"DIR::{biome}::{R}::{time.time_ns()}"
    start_pos = LOG.mark()

    _send_direction_vote(biome, R, token)
    LOG.wait_for_token(token, start_pos=start_pos, timeout=TOKEN_TIMEOUT)

    with open(LATEST_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(start_pos)
        chunk = f.read(80000)

    mx = re.findall(r"\[CHAT\].* has (-?\d+) \[dirX\]", chunk)
    mz = re.findall(r"\[CHAT\].* has (-?\d+) \[dirZ\]", chunk)
    if not mx or not mz:
        return (0, 0)

    dirx = int(mx[-1])
    dirz = int(mz[-1])

    sx = 0 if dirx == 0 else (1 if dirx > 0 else -1)
    sz = 0 if dirz == 0 else (1 if dirz > 0 else -1)
    return (sx, sz)

# =========================
# FAST HILLCLIMB (few steps)
# =========================
def hillclimb_to_safe_spot(biome: str) -> bool:
    R = R_CHECK
    steps = 0
    moves = 0
    last_dir = None
    reverse_hits = 0
    ambiguous_hits = 0

    while True:
        if steps >= MAX_STEPS or moves >= GLOBAL_MAX_MOVES:
            return False

        try:
            if is_safe_in_biome(biome, R):
                return True
        except TimeoutError:
            # log geciktiyse bu tur safe sayma
            pass

        try:
            sx, sz = compute_direction(biome, R)
        except TimeoutError:
            sx, sz = (0, 0)

        if sx == 0 and sz == 0:
            ambiguous_hits += 1
            if ambiguous_hits <= DIR_ZERO_FALLBACK_TRIES:
                nudge_random_small()
                steps += 1
                moves += 1
                continue
            return False

        # ping-pong engeli (ters yöne dönme)
        if last_dir is not None and (sx, sz) == (-last_dir[0], -last_dir[1]):
            reverse_hits += 1
            if reverse_hits <= 1:
                nudge_random_small()
                steps += 1
                moves += 1
                continue
            return False

        dx = sx * MOVE_STEP
        dz = sz * MOVE_STEP
        run_command(f"/tp ~{dx} ~ ~{dz}", wait_after=0.1)
        time.sleep(TP_SETTLE_WAIT)

        last_dir = (sx, sz)
        steps += 1
        moves += 1

# =========================
# SCREENSHOT
# =========================
def position_for_screenshot():
    run_command("/tp ~ ~100 ~", wait_after=WAIT_AFTER_Y_MOVE)
    press_double_space()
    time.sleep(WAIT_FALL)
    press_double_space()
    time.sleep(0.5)
    run_command(f"/tp ~ ~{SS_HEIGHT} ~", wait_after=WAIT_AFTER_Y_MOVE)

def take_screenshot(biome: str, cycle_index: int):
    biome_dir = os.path.join(BASE_DIR, biome)
    os.makedirs(biome_dir, exist_ok=True)

    pyautogui.press('f1')
    time.sleep(0.6)

    filename = os.path.join(biome_dir, f"{cycle_index}.png")
    pyautogui.screenshot().save(filename)

    time.sleep(0.3)
    pyautogui.press('f1')

# =========================
# MAIN LOOP
# =========================
def process_biome(biome: str, cycle_index: int):
    biome = biome.strip()
    if not biome:
        return

    if not go_to_biome(biome):
        return

    if not hillclimb_to_safe_spot(biome):
        print(f"[SKIP] {biome}: safe spot bulunamadı (hız modu).")
        return

    position_for_screenshot()
    take_screenshot(biome, cycle_index)

def main():
    ensure_scoreboards_once()

    with open("biomes.txt", "r", encoding="utf-8") as f:
        biomes = [line.strip() for line in f if line.strip()]

    cycle_index = 0
    while True:
        for biome in biomes:
            try:
                process_biome(biome, cycle_index)
            except Exception as e:
                print(f"[ERROR] biome={biome} cycle={cycle_index}: {e}")

        # Tüm biomes turu bitti -> ekstra çeşitlilik için ileri atla
        move_30000_x()

        cycle_index += 1
        if cycle_index > LOOP_MAX_ITERATION:
            break

if __name__ == "__main__":
    try:
        main()
    finally:
        LOG.close()
