import cv2
import numpy as np
from mss import mss

# =========================
# Generic selection function
# =========================
def select_rectangle(img, message):
    drawing = False
    done = False
    ix, iy = -1, -1
    fx, fy = -1, -1
    img_copy = img.copy()

    def draw(event, x, y, flags, param):
        nonlocal ix, iy, fx, fy, drawing, done, img

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img = img_copy.copy()
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            fx, fy = x, y
            done = True

    # Fullscreen window
    cv2.namedWindow(message, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(message, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(message, draw)

    while True:
        cv2.imshow(message, img)

        if done:
            break

        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()

    # Normalize coordinates
    x1, y1 = min(ix, fx), min(iy, fy)
    x2, y2 = max(ix, fx), max(iy, fy)

    return x1, y1, x2, y2


# =========================
# Initial screenshot
# =========================
with mss() as sct:
    monitor = sct.monitors[1]
    screenshot = sct.grab(monitor)

img = np.array(screenshot)
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# =========================
# STEP 1 - Page selection
# =========================
print("\nSTEP 1: Select the PAGE area (drag the mouse)")

x1, y1, x2, y2 = select_rectangle(img.copy(), "Select PAGE AREA")

page_vertices = [
    (x1, y1),
    (x2, y1),
    (x2, y2),
    (x1, y2)
]

print("\nPage vertices:")
for v in page_vertices:
    print(v)

# =========================
# STEP 2 - Symbol selection
# =========================
print("\nSTEP 2: Select the SYMBOL (drag the mouse)")

sx1, sy1, sx2, sy2 = select_rectangle(img.copy(), "Select SYMBOL")

# Compute center
cx = (sx1 + sx2) // 2
cy = (sy1 + sy2) // 2

print("\nSymbol center:")
print((cx, cy))

# =========================
# FINAL RESULT
# =========================
print("\n--- RESULT ---")
print("Page area (vertices):", page_vertices)
print("Symbol center:", (cx, cy))
