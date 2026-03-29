import cv2
import numpy as np
from mss import mss

drawing = False
done = False
ix, iy = -1, -1
fx, fy = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, img, img_copy, done

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

        # Normalizza coordinate
        x1, y1 = min(ix, fx), min(iy, fy)
        x2, y2 = max(ix, fx), max(iy, fy)

        vertices = [
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2)
        ]

        print("\nVertici selezionati:")
        for v in vertices:
            print(v)

        done = True  # 👈 fa uscire dal loop

# Screenshot
with mss() as sct:
    monitor = sct.monitors[1]
    screenshot = sct.grab(monitor)

img = np.array(screenshot)
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
img_copy = img.copy()

# Fullscreen
cv2.namedWindow("Seleziona area", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Seleziona area", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.setMouseCallback("Seleziona area", draw_rectangle)

while True:
    cv2.imshow("Seleziona area", img)

    if done:
        break  # 👈 esce subito quando hai finito

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()