import cv2
import hashlib
import numpy as np
import os
import pyautogui
import pywinctl as pwc
import sys
import time
from io import BytesIO
from mss import mss
from PIL import Image

ICON_FILE = "arrow.png"   # <-- your icon file
SCREENSHOT_DIR = "screenshots"

# pixels to trim from each side
TRIM_TOP = 140
TRIM_BOTTOM = 90
TRIM_LEFT = 160
TRIM_RIGHT = 100

pdf_images = []

def file_md5(path):
    """Compute the MD5 of the given file."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def save_screenshot(screenshot):
    """Save the image with an incremental name inside SCREENSHOT_DIR."""
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
    print(f"Screenshot saved as {path}")

    return path


def get_virtualbox_window():
    all_windows = pwc.getAllWindows()
    wins = [w for w in all_windows if "VirtualBox" in w.title]

    if not wins:
        raise RuntimeError("VirtualBox window not found.")

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
        raise RuntimeError("Unable to load icon: " + icon_file)

    res = cv2.matchTemplate(screenshot, icon, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    print("Match:", max_val)

    if max_val < 0.75:  # match threshold
        del icon
        return None

    w, h = icon.shape[1], icon.shape[0]
    del icon
    return max_loc, w, h

def image_md5_small(image_np):
    small = cv2.resize(image_np, (64, 64))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return hashlib.md5(gray.tobytes()).hexdigest()

def add_to_pdf(image_np):
    # Convert numpy → PIL (grayscale consigliato)
    img = Image.fromarray(image_np).convert("L")

    # Comprimi in JPEG in memoria
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=50)
    buffer.seek(0)

    # Riapri come immagine PIL (necessario per PDF)
    compressed_img = Image.open(buffer)

    pdf_images.append(compressed_img)

def process_page(prev_md5=None):
    win = get_virtualbox_window()
    screenshot, win_x, win_y = screenshot_window(win)

    # trim borders
    height, width, _ = screenshot.shape
    screenshot_trimmed = screenshot[
        TRIM_TOP:height - TRIM_BOTTOM,
        TRIM_LEFT:width - TRIM_RIGHT
    ]

    add_to_pdf(screenshot_trimmed)

    # compute page md5
    current_md5 = image_md5_small(screenshot_trimmed)

    # compare with previous page
    if prev_md5 and current_md5 == prev_md5:
        print("Page identical to the previous one. End of document.")
        return False, prev_md5

    found = find_icon_in_image(screenshot, ICON_FILE)
    if not found:
        print("Icon not found in the window.")
        return False, current_md5

    (px, py), w, h = found
    click_x = win_x + px + w // 2
    click_y = win_y + py + h // 2

    print("Clicking at:", click_x, click_y)
    pyautogui.click(click_x, click_y)
    print("Icon clicked!")

    return True, current_md5

def save_pdf(output_path):
    if not pdf_images:
        print("No images to save.")
        return

    pdf_images[0].save(
        output_path,
        save_all=True,
        append_images=pdf_images[1:]
    )

    print(f"PDF created: {output_path}")

def main():
    output_name = sys.argv[1] if len(sys.argv) >= 2 else "output.pdf"

    prev_md5 = None
    page_count = 0

    print("Starting capture...")

    while True:
        success, prev_md5 = process_page(prev_md5)

        if not success:
            break

        page_count += 1
        print(f"Captured page {page_count}")

        time.sleep(3.0)

    print(f"Finished. Total pages: {page_count}")

    save_pdf(output_name)

if __name__ == "__main__":
    main()

