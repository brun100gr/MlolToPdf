import cv2
import numpy as np
from mss import mss

# =========================
# Funzione generica selezione
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

    # Finestra fullscreen
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

    # Normalizza coordinate
    x1, y1 = min(ix, fx), min(iy, fy)
    x2, y2 = max(ix, fx), max(iy, fy)

    return x1, y1, x2, y2


# =========================
# Screenshot iniziale
# =========================
with mss() as sct:
    monitor = sct.monitors[1]
    screenshot = sct.grab(monitor)

img = np.array(screenshot)
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# =========================
# STEP 1 - Selezione pagina
# =========================
print("\nSTEP 1: Seleziona l'area della PAGINA (trascina il mouse)")

x1, y1, x2, y2 = select_rectangle(img.copy(), "Seleziona AREA PAGINA")

page_vertices = [
    (x1, y1),
    (x2, y1),
    (x2, y2),
    (x1, y2)
]

print("\nVertici pagina:")
for v in page_vertices:
    print(v)

# =========================
# STEP 2 - Selezione simbolo
# =========================
print("\nSTEP 2: Seleziona il SIMBOLO (trascina il mouse)")

sx1, sy1, sx2, sy2 = select_rectangle(img.copy(), "Seleziona SIMBOLO")

# Calcolo centro
cx = (sx1 + sx2) // 2
cy = (sy1 + sy2) // 2

print("\nCentro simbolo:")
print((cx, cy))

# =========================
# RISULTATO FINALE
# =========================
print("\n--- RISULTATO ---")
print("Area pagina (vertici):", page_vertices)
print("Centro simbolo:", (cx, cy))
