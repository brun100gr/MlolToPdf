import os
import cv2
import sys
import numpy as np
import pyautogui
import pywinctl as pwc
from mss import mss
from PIL import Image
import hashlib
import time

ICON_FILE = "arrow.png"   # <-- il tuo file icona
SCREENSHOT_DIR = "screenshots"

# pixel da tagliare per ciascun lato
TRIM_TOP = 140
TRIM_BOTTOM = 90
TRIM_LEFT = 160
TRIM_RIGHT = 100


def file_md5(path):
    """Calcola l'MD5 del file indicato."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def save_screenshot(screenshot):
    """Salva l'immagine con nome progressivo nella cartella SCREENSHOT_DIR."""
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    existing_files = [
        f for f in os.listdir(SCREENSHOT_DIR)
        if f.startswith("page_") and f.endswith(".png")
    ]
    existing_numbers = [int(f[5:9]) for f in existing_files if f[5:9].isdigit()]
    next_number = max(existing_numbers, default=0) + 1

    filename = f"page_{next_number:04d}.png"
    path = os.path.join(SCREENSHOT_DIR, filename)

    screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, screenshot_bgr)
    print(f"Screenshot salvato come {path}")

    return path


def get_virtualbox_window():
    all_windows = pwc.getAllWindows()
    wins = [w for w in all_windows if "VirtualBox" in w.title]

    if not wins:
        raise RuntimeError("Finestra VirtualBox non trovata.")

    return wins[0]


def screenshot_window(win):
    x, y, w, h = win.left, win.top, win.width, win.height

    with mss() as sct:
        raw = sct.grab({
            "left": x,
            "top": y,
            "width": w,
            "height": h
        })

    img = Image.frombytes("RGB", (raw.width, raw.height), raw.rgb)
    screenshot_np = np.array(img)
    del img
    del raw
    return screenshot_np, x, y


def find_icon_in_image(screenshot, icon_file):
    icon = cv2.imread(icon_file)
    if icon is None:
        raise RuntimeError("Impossibile caricare l’icona: " + icon_file)

    res = cv2.matchTemplate(screenshot, icon, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    print("Match:", max_val)

    if max_val < 0.75:  # soglia di match
        del icon
        return None

    w, h = icon.shape[1], icon.shape[0]
    del icon
    return max_loc, w, h


def process_page(prev_md5=None):
    win = get_virtualbox_window()
    screenshot, win_x, win_y = screenshot_window(win)

    # ritaglio dei bordi
    height, width, _ = screenshot.shape
    screenshot_trimmed = screenshot[
        TRIM_TOP:height - TRIM_BOTTOM,
        TRIM_LEFT:width - TRIM_RIGHT
    ]

    # salva screenshot e calcola md5
    path = save_screenshot(screenshot_trimmed)
    current_md5 = file_md5(path)

    # confronta con la pagina precedente
    if prev_md5 and current_md5 == prev_md5:
        print("Pagina identica alla precedente. Fine documento.")
        os.remove(path)  # elimina l’ultima immagine duplicata
        return False, prev_md5

    found = find_icon_in_image(screenshot, ICON_FILE)
    if not found:
        print("Icona non trovata nella finestra.")
        return False, current_md5

    (px, py), w, h = found
    click_x = win_x + px + w // 2
    click_y = win_y + py + h // 2

    print("Click su:", click_x, click_y)
    pyautogui.click(click_x, click_y)
    print("Icona cliccata!")

    return True, current_md5

def images_to_pdf(folder_path, output_pdf="output.pdf"):
    # Prende tutti i file PNG nella cartella
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]

    if not files:
        print("Nessuna immagine PNG trovata nella cartella.")
        return

    # Ordina i file in ordine alfabetico (page_0001.png, page_0002.png, ecc.)
    files.sort()

    images = []
    for filename in files:
        path = os.path.join(folder_path, filename)
        img = Image.open(path).convert("RGB")
        images.append(img)

    # Salva tutte le immagini in un unico PDF
    output_path = os.path.join(folder_path, output_pdf)
    images[0].save(output_path, save_all=True, append_images=images[1:])
    print(f"PDF creato con successo: {output_path}")

def main():
    # Create pdf from all images
    folder = sys.argv[1]
    output_name = sys.argv[2] if len(sys.argv) >= 3 else "output.pdf"

    images_to_pdf(folder, output_name)
 
if __name__ == "__main__":
    main()

