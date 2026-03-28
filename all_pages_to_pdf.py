import cv2
import hashlib
import numpy as np
import pyautogui
import pywinctl as pwc
import sys
import time
from io import BytesIO
from mss import mss
from PIL import Image

ICON_FILE = "arrow.png"   # <-- your icon file

# Pixels to trim from each side
TRIM_TOP = 140
TRIM_BOTTOM = 90
TRIM_LEFT = 160
TRIM_RIGHT = 100

pdf_images = []

def rgb_hex_to_hsv(hex_color):
    """Converte un colore RGB esadecimale (es: 'ffff13') in HSV OpenCV."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    rgb = np.uint8([[[r, g, b]]])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return hsv[0][0]

# Esempio d'uso:
hsv = rgb_hex_to_hsv("ffff13")
print(hsv)  # Output: [ 30 236 255]

def get_virtualbox_window():
    """Find the VirtualBox window."""
    all_windows = pwc.getAllWindows()
    wins = [w for w in all_windows if "VirtualBox" in w.title]

    if not wins:
        raise RuntimeError("VirtualBox window not found.")

    return wins[0]

def screenshot_window(win):
    """Take a screenshot of the given window."""
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
    """Locate the icon inside the screenshot using template matching."""
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

def has_orange_box(image_np):
    """Detect orange UI highlight (even if not a closed rectangle)."""
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # Dopo la conversione in HSV l'arancione #FFA500 dibenta giallo FFFF13
    lower_yellow = np.array([18, 255, 255])
    upper_yellow = np.array([20, 255, 255])

    # Dopo la conversione in HSV
    cv2.imwrite(f"debug_hsv_{int(time.time()*1000)}.png", hsv)
    print("HSV pixel (189,719):", hsv[189, 719])  # Cambia con coordinate interne al rettangolo

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    cv2.imwrite(f"mask_debug_{int(time.time()*1000)}.png", mask)

    # Remove noise - there is no need for this step, no noise present
    #kernel = np.ones((3, 3), np.uint8)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(f"mask_debug_{int(time.time()*1000)}.png", mask)  # Salva la maschera con timestamp

    # Count orange pixels
    yellow_hvs_pixels = np.sum(mask > 0)

    if yellow_hvs_pixels > 0:
        print("----> yellow_hvs_pixels:", yellow_hvs_pixels)

    if yellow_hvs_pixels < 200:
        return False  # too small → ignore

    # Detect long lines (edges of rectangle)
    edges = cv2.Canny(mask, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=10
    )

    if lines is None:
        return False

    # Count long lines
    long_lines = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)

        if length > 80:
            long_lines += 1

    print("long_lines:", long_lines)

    # If we detect at least 2-3 long lines → likely the rectangle
    return long_lines >= 2

def wait_until_clean(win):
    while True:
        screenshot, win_x, win_y = screenshot_window(win)

        height, width, _ = screenshot.shape
        screenshot_trimmed = screenshot[
            TRIM_TOP:height - TRIM_BOTTOM,
            TRIM_LEFT:width - TRIM_RIGHT
        ]

        if not has_orange_box(screenshot_trimmed):
            return screenshot, win_x, win_y

        print("Orange box detected, waiting...")
        time.sleep(0.1)

def image_md5_small(image_np):
    """Compute an MD5 hash of a small grayscale version of the image."""
    small = cv2.resize(image_np, (64, 64))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return hashlib.md5(gray.tobytes()).hexdigest()

def add_to_pdf(image_np):
    """Convert a numpy array to a PIL image, compress it, and add it to the PDF list."""
    img = Image.fromarray(image_np).convert("L")  # grayscale
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=50)
    buffer.seek(0)
    pdf_images.append(Image.open(buffer))

def process_page(prev_md5=None, page_num=1):
    """Capture a page, add it to PDF, and click the icon to go to the next page."""
    win = get_virtualbox_window()
    # Wait until the page is clean (no orange box)
    screenshot, win_x, win_y = wait_until_clean(win)
    cv2.imwrite(f"screenshot_{page_num}.png", cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))

    # Trim borders
    height, width, _ = screenshot.shape
    screenshot_trimmed = screenshot[
        TRIM_TOP:height - TRIM_BOTTOM,
        TRIM_LEFT:width - TRIM_RIGHT
    ]

    # Salva l'immagine su file
    cv2.imwrite(f"screenshot_trimmed_{page_num}.png", cv2.cvtColor(screenshot_trimmed, cv2.COLOR_RGB2BGR))

    add_to_pdf(screenshot_trimmed)

    # Compute page MD5
    current_md5 = image_md5_small(screenshot_trimmed)

    # Compare with previous page
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
    """Save all captured images into a single PDF file."""
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
        success, prev_md5 = process_page(prev_md5, page_count + 1)

        if not success:
            break

        page_count += 1
        print(f"Captured page {page_count}")

        time.sleep(1.3)

    print(f"Finished. Total pages: {page_count}")

    save_pdf(output_name)

if __name__ == "__main__":
    main()