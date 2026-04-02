"""
mlol_to_pdf.py — Automatic capture of pages from a VirtualBox window and save to PDF.

Usage:
    # Interactive selection of page area and icon (recommended on first run):
    python mlol_to_pdf.py book.pdf --select

    # Re-use previously selected coordinates (faster for subsequent runs):
    python mlol_to_pdf.py book.pdf --page-area 160 140 1760 1010 --icon-center 1840 950

    # Additional options:
    python mlol_to_pdf.py book.pdf --select --delay 0.4 --quality 50 --split 200 --mode bw

  --select          Interactively select the page area and icon centre before
                    starting. Launches a fullscreen overlay.
  --page-area X1 Y1 X2 Y2
                    Absolute screen coordinates of the page content area
                    (top-left and bottom-right corners).
  --icon-center X Y
                    Absolute screen coordinates of the centre of the
                    "next page" icon to click.
  --split N         After saving, split the PDF into chunks of N pages.
  --mode MODE       Page compression: jpeg (default), jpeg-hq, bw.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pyautogui
import pywinctl as pwc
from mss import mss
from PIL import Image
from pypdf import PdfReader, PdfWriter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    Central configuration object. All tunable parameters live here so that
    no magic numbers are scattered throughout the code.
    """
    output_path: str = "output.pdf"

    # Absolute screen coordinates of the page content area (set interactively
    # via --select or explicitly via --page-area). Used to crop each screenshot
    # to only the document content, replacing the old fixed trim_* offsets.
    page_x1: int = 0
    page_y1: int = 0
    page_x2: int = 0
    page_y2: int = 0

    # Absolute screen coordinates of the centre of the "next page" icon.
    # pyautogui clicks this point directly, so no template matching is needed.
    icon_cx: int = 0
    icon_cy: int = 0

    # Minimum confidence score (0-1) for template matching to accept a hit.
    match_threshold: float = 0.75

    # Minimum number of orange/yellow HSV pixels to consider an orange box present.
    orange_pixel_min: int = 200

    # A detected line must be longer than this (pixels) to count as a real edge.
    line_length_min: int = 80

    # How many long lines must be found to declare an orange box present.
    long_lines_min: int = 2

    # Seconds to wait between clicking the icon and capturing the next page.
    delay: float = 0.4

    # Safety cap: stop after this many pages even if the end is not detected.
    max_pages: int = 2000



    # When True, intermediate debug images are written to disk.
    debug: bool = False

    # Seconds to wait after selection before starting the capture loop,
    # giving the user time to bring the target window to the foreground.
    start_delay: int = 2

    # If > 0, the final PDF is split into chunks of this many pages.
    split_pages: int = 0


# ---------------------------------------------------------------------------
# Interactive area selection (Tkinter + PIL screenshot background)
# ---------------------------------------------------------------------------

def _run_tk_selector(screenshot_pil, message: str) -> tuple[int, int, int, int]:
    """
    Show a fullscreen Tkinter window with the screenshot as background.
    The user drags a rectangle; outside the selection is darkened (Ubuntu style).
    Returns (x1, y1, x2, y2). Pressing Esc exits the program.
    """
    import tkinter as tk
    from PIL import ImageDraw, ImageTk

    # Pre-compute the darkened base image (selection area will be shown bright).
    dark_overlay = Image.new("RGBA", screenshot_pil.size, (0, 0, 0, 120))
    base_dark = Image.alpha_composite(
        screenshot_pil.convert("RGBA"), dark_overlay
    ).convert("RGB")

    result: list[tuple[int,int,int,int]] = []

    root = tk.Tk()
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    print(f"  Tkinter screen size: {sw}x{sh}")
    root.geometry(f"{sw}x{sh}+0+0")
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.overrideredirect(True)

    # Resize screenshot to match actual screen size if they differ.
    if screenshot_pil.width != sw or screenshot_pil.height != sh:
        print(f"  Resizing screenshot from {screenshot_pil.width}x{screenshot_pil.height} to {sw}x{sh}")
        screenshot_pil = screenshot_pil.resize((sw, sh), Image.LANCZOS)
        base_dark = Image.alpha_composite(
            screenshot_pil.convert("RGBA"), Image.new("RGBA", screenshot_pil.size, (0, 0, 0, 120))
        ).convert("RGB")

    tk_base = ImageTk.PhotoImage(base_dark)
    canvas = tk.Canvas(root, cursor="cross", highlightthickness=0,
                       width=sw, height=sh)
    canvas.pack(fill=tk.BOTH, expand=True)
    bg_item = canvas.create_image(0, 0, anchor="nw", image=tk_base)

    # Instruction banner
    w = screenshot_pil.width
    canvas.create_rectangle(0, 0, w, 70, fill="black", outline="")
    canvas.create_text(20, 14, anchor="nw", text=message,
                       fill="white", font=("Arial", 22, "bold"))
    canvas.create_text(20, 46, anchor="nw",
                       text="Drag to select. Release to confirm. Esc = cancel.",
                       fill="yellow", font=("Arial", 13))

    state = {"sx": 0, "sy": 0, "tk_current": None}

    def redraw(x1, y1, x2, y2):
        bright_crop = screenshot_pil.crop((x1, y1, x2, y2))
        composite = base_dark.copy()
        composite.paste(bright_crop, (x1, y1))
        from PIL import ImageDraw as ID
        ID.Draw(composite).rectangle([x1, y1, x2-1, y2-1], outline="red", width=2)
        state["tk_current"] = ImageTk.PhotoImage(composite)
        canvas.itemconfig(bg_item, image=state["tk_current"])

    def on_press(e):
        state["sx"], state["sy"] = e.x, e.y

    def on_drag(e):
        x1, y1 = min(state["sx"], e.x), min(state["sy"], e.y)
        x2, y2 = max(state["sx"], e.x), max(state["sy"], e.y)
        if x2 > x1 and y2 > y1:
            redraw(x1, y1, x2, y2)

    def on_release(e):
        x1, y1 = min(state["sx"], e.x), min(state["sy"], e.y)
        x2, y2 = max(state["sx"], e.x), max(state["sy"], e.y)
        result.append((x1, y1, x2, y2))
        root.quit()

    def on_esc(e):
        print("Selection cancelled.")
        root.quit()
        root.destroy()
        sys.exit(0)

    canvas.bind("<ButtonPress-1>",   on_press)
    canvas.bind("<B1-Motion>",       on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", on_esc)

    root.mainloop()
    root.destroy()
    return result[0]


def run_interactive_selection() -> tuple[tuple[int,int,int,int], tuple[int,int]]:
    """
    Capture a screenshot of the whole screen, then ask the user to:
      1. Draw a rectangle around the page content area.
      2. Draw a rectangle around the "next page" icon.

    Returns:
        page_area   - (x1, y1, x2, y2) absolute screen coordinates of the page.
        icon_center - (cx, cy) absolute screen coordinates of the icon centre.
    """
    # Take the screenshot BEFORE opening any window to avoid capturing our own UI.
    print("Taking screenshot for selection overlay...")
    with mss() as sct:
        # monitors[0] is the virtual bounding box covering ALL monitors combined.
        # monitors[1], [2], ... are individual screens. Using [0] lets the user
        # select areas on any monitor, not just the primary one.
        monitor = sct.monitors[0]
        raw = sct.grab(monitor)
        screenshot_pil = Image.frombytes("RGB", (raw.width, raw.height), raw.rgb)
    print(f"  Screenshot size: {screenshot_pil.width}x{screenshot_pil.height} (all monitors)")

    # --- Step 1: page area ---
    print("\nSTEP 1: Draw a rectangle around the PAGE CONTENT AREA, then release.")
    x1, y1, x2, y2 = _run_tk_selector(screenshot_pil, "STEP 1 — Select PAGE AREA")
    print(f"  Page area: ({x1}, {y1}) -> ({x2}, {y2})")

    # --- Step 2: icon ---
    print("\nSTEP 2: Draw a rectangle around the NEXT-PAGE ICON, then release.")
    sx1, sy1, sx2, sy2 = _run_tk_selector(screenshot_pil, "STEP 2 — Select NEXT-PAGE ICON")
    cx = (sx1 + sx2) // 2
    cy = (sy1 + sy2) // 2
    print(f"  Icon centre: ({cx}, {cy})")

    print("\n--- Selection complete ---")
    print(f"  --page-area {x1} {y1} {x2} {y2} --icon-center {cx} {cy}")
    print("  (You can paste the line above to skip selection next time.)\n")

    return (x1, y1, x2, y2), (cx, cy)


# ---------------------------------------------------------------------------
# Screenshot and crop
# ---------------------------------------------------------------------------

def screenshot_fullscreen() -> np.ndarray:
    """
    Capture the entire primary monitor and return it as an RGB numpy array.
    Used each capture cycle to grab the current page.
    """
    with mss() as sct:
        # Use monitors[0] (all monitors combined) so that coordinates selected
        # interactively on any monitor match the capture coordinate space.
        monitor = sct.monitors[0]
        raw = sct.grab(monitor)
        img = np.frombuffer(raw.rgb, dtype=np.uint8).reshape((raw.height, raw.width, 3))
    return img.copy()


def crop_to_page(screenshot: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Crop the full-screen screenshot to the page content area selected by the
    user. Uses absolute screen coordinates stored in cfg, replacing the old
    fixed trim_* pixel offsets.
    """
    return screenshot[cfg.page_y1:cfg.page_y2, cfg.page_x1:cfg.page_x2]


# ---------------------------------------------------------------------------
# Orange box detection
# ---------------------------------------------------------------------------

def has_orange_box(image_np: np.ndarray, cfg: Config, debug_suffix: str = "") -> bool:
    """
    Detect the orange VirtualBox selection highlight in the page area.

    Strategy:
      1. Convert to HSV. Orange #FFA500 -> hue ~18-25 in OpenCV HSV.
      2. Threshold to a binary mask of orange-ish pixels.
      3. Early-exit if pixel count is below minimum.
      4. Run Canny + HoughLinesP: a real rectangle produces long straight edges.
    """
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower = np.array([18, 200, 200])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    if cfg.debug and debug_suffix:
        cv2.imwrite(f"debug_hsv_{debug_suffix}.png", hsv)
        cv2.imwrite(f"debug_mask_{debug_suffix}.png", mask)

    yellow_pixels = int(np.sum(mask > 0))
    if yellow_pixels < cfg.orange_pixel_min:
        return False

    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=50, minLineLength=50, maxLineGap=10,
    )
    if lines is None:
        return False

    long_lines = sum(
        1 for line in lines
        if np.hypot(line[0][2] - line[0][0], line[0][3] - line[0][1]) > cfg.line_length_min
    )

    if cfg.debug:
        print(f"  [debug] yellow_pixels={yellow_pixels}, long_lines={long_lines}")

    return long_lines >= cfg.long_lines_min


def wait_until_clean(cfg: Config, page_num: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Poll the screen repeatedly until the orange selection box disappears from
    the page area. Returns (full_screenshot, cropped_page) once clean.
    """
    while True:
        screenshot = screenshot_fullscreen()
        page = crop_to_page(screenshot, cfg)

        if not has_orange_box(page, cfg, debug_suffix=str(page_num) if cfg.debug else ""):
            return screenshot, page

        print("  Orange box detected, waiting...")
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# Page hashing
# ---------------------------------------------------------------------------

def image_md5(image_np: np.ndarray, size: int = 128) -> str:
    """
    Compute a perceptual fingerprint by downscaling to 128x128 grayscale.
    Used to detect when the document has looped back (= end of book).
    """
    small = cv2.resize(image_np, (size, size))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return hashlib.md5(gray.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Page compression
# ---------------------------------------------------------------------------


def compress_page(image_np: np.ndarray) -> bytes:
    """
    Convert a page image to lossless PNG bytes.
    PNG is lossless so there are no JPEG compression artefacts on text.
    Returns raw bytes so the buffer can be immediately garbage-collected.
    """
    img = Image.fromarray(image_np)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Page capture
# ---------------------------------------------------------------------------

def process_page(
    cfg: Config,
    prev_md5: str | None,
    page_num: int,
) -> tuple[bool, str | None, bytes | None]:
    """
    Capture one page and click the icon to advance to the next.

    Steps:
      1. Wait for the orange box to clear (page transition finished).
      2. Crop the screenshot to the page area (coordinates from interactive
         selection, not fixed trim offsets).
      3. Hash and compare with previous page to detect end-of-document,
         with up to 2 retries to rule out slow transitions.
      4. Compress the page according to cfg.mode.
      5. Click the pre-selected icon centre to advance.

    Returns (success, current_md5, page_bytes).
    """
    screenshot, page = wait_until_clean(cfg, page_num)

    if cfg.debug:
        cv2.imwrite(f"debug_page_{page_num}.png", cv2.cvtColor(page, cv2.COLOR_RGB2BGR))

    current_md5 = image_md5(page)
    print(f"  MD5: {current_md5[:12]}...  prev: {prev_md5[:12] if prev_md5 else 'None'}")

    # End-of-document check with retries to tolerate slow page transitions.
    if prev_md5 and current_md5 == prev_md5:
        MAX_RETRIES = 2
        for attempt in range(1, MAX_RETRIES + 1):
            print(f"  Page identical to previous — waiting {cfg.delay}s before retry {attempt}/{MAX_RETRIES}...")
            time.sleep(cfg.delay)
            screenshot, page = wait_until_clean(cfg, page_num)
            current_md5 = image_md5(page)
            if current_md5 != prev_md5:
                print(f"  Page changed after retry {attempt} — continuing.")
                break
        else:
            print("  Page still identical after all retries — end of document.")
            return False, prev_md5, None

    page_bytes = compress_page(page)

    # Click the icon at the pre-selected absolute screen coordinates.
    print(f"  Click at ({cfg.icon_cx}, {cfg.icon_cy})")
    pyautogui.click(cfg.icon_cx, cfg.icon_cy)

    return True, current_md5, page_bytes


# ---------------------------------------------------------------------------
# PDF splitting
# ---------------------------------------------------------------------------

def split_pdf(output_path: str, step: int) -> None:
    """
    Split a PDF into equal-sized chunks using pypdf (pure Python).

    Naming convention:
      <base>_0001_0200.pdf  - full chunk
      <base>_0201_end.pdf   - final partial chunk
    """
    base = Path(output_path).stem
    out_dir = Path(output_path).parent

    reader = PdfReader(output_path)
    total = len(reader.pages)
    print(f"  Total pages in PDF: {total}")

    start = 0
    block_start_label = 1

    while start < total:
        end = min(start + step, total)
        end_label = start + step
        is_last = end < start + step

        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        if is_last:
            out_file = out_dir / f"{base}_{block_start_label:04d}_end.pdf"
        else:
            out_file = out_dir / f"{base}_{block_start_label:04d}_{end_label:04d}.pdf"

        print(f"  Pages {block_start_label}-{end} -> {out_file.name}")
        with open(out_file, "wb") as f:
            writer.write(f)

        start += step
        block_start_label += step

    print("Splitting completed.")


# ---------------------------------------------------------------------------
# PDF saving
# ---------------------------------------------------------------------------

def save_pdf(pages_bytes: list[bytes], output_path: str) -> None:
    """
    Assemble per-page PNG bytes into a single PDF using Pillow.
    PNG is lossless so text quality is preserved exactly as captured.
    """
    if not pages_bytes:
        print("No images to save.")
        return

    images = [Image.open(BytesIO(b)) for b in pages_bytes]
    images[0].save(
        output_path,
        format="PDF",
        save_all=True,
        append_images=images[1:],
    )
    size_kb = Path(output_path).stat().st_size // 1024
    print(f"PDF saved: {output_path}  ({size_kb} KB, {len(pages_bytes)} pages)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Capture pages from VirtualBox and save them as PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output", nargs="?", default="output.pdf",
                        help="Output PDF file (default: output.pdf)")

    # --- Area selection ---
    sel_group = parser.add_mutually_exclusive_group(required=True)
    sel_group.add_argument("--select", action="store_true",
                           help="Interactively select the page area and icon before starting")
    sel_group.add_argument("--page-area", nargs=4, type=int, metavar=("X1","Y1","X2","Y2"),
                           help="Absolute screen coords of the page content area")

    parser.add_argument("--icon-center", nargs=2, type=int, metavar=("X","Y"),
                        help="Absolute screen coords of the icon centre (required with --page-area)")

    # --- Capture options ---
    parser.add_argument("--delay", type=float, default=0.4,
                        help="Seconds to wait after clicking before capturing the next page")
    parser.add_argument("--max-pages", type=int, default=2000,
                        help="Stop after this many pages even if end is not detected")
    parser.add_argument("--split", type=int, default=0, metavar="N",
                        help="Split output PDF into chunks of N pages (0 = disabled)")
    parser.add_argument("--start-delay", type=int, default=2, metavar="N",
                        help="Countdown seconds after Enter before capture starts (default: 2)")
    parser.add_argument("--debug", action="store_true",
                        help="Write intermediate debug images to disk")

    args = parser.parse_args()

    # Validate: --icon-center is required when --page-area is used
    if args.page_area and not args.icon_center:
        parser.error("--icon-center is required when using --page-area")

    # Resolve page area and icon centre
    if args.select:
        page_area, icon_center = run_interactive_selection()
    else:
        page_area = tuple(args.page_area)
        icon_center = tuple(args.icon_center)

    output_path = args.output
    if not output_path.lower().endswith(".pdf"):
        output_path += ".pdf"

    return Config(
        output_path=output_path,
        page_x1=page_area[0],
        page_y1=page_area[1],
        page_x2=page_area[2],
        page_y2=page_area[3],
        icon_cx=icon_center[0],
        icon_cy=icon_center[1],
        delay=args.delay,
        max_pages=args.max_pages,
        split_pages=args.split,
        start_delay=args.start_delay,
        debug=args.debug,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = parse_args()

    print(f"Page area:   ({cfg.page_x1}, {cfg.page_y1}) -> ({cfg.page_x2}, {cfg.page_y2})")
    print(f"Icon centre: ({cfg.icon_cx}, {cfg.icon_cy})")
    # Wait for the user to bring the target window to the foreground,
    # then start a short countdown so they have time to release the keyboard.
    input("\nBring the target window to the foreground, then press Enter to start...")
    if cfg.start_delay > 0:
        print(f"Starting in {cfg.start_delay}s...")
        for remaining in range(cfg.start_delay, 0, -1):
            print(f"  {remaining}...", end="\r", flush=True)
            time.sleep(1)
        print("  Go!             ")

    print("Starting capture...\n")

    pages: list[bytes] = []
    prev_md5: str | None = None
    page_num = 0

    while page_num < cfg.max_pages:
        page_num += 1
        print(f"Page {page_num}...")

        success, prev_md5, page_bytes = process_page(cfg, prev_md5, page_num)

        if page_bytes is not None:
            pages.append(page_bytes)

        if not success:
            break

        time.sleep(cfg.delay)

    print(f"\nCapture completed. Total pages: {len(pages)}")
    save_pdf(pages, cfg.output_path)

    if cfg.split_pages > 0:
        print(f"\nSplitting into chunks of {cfg.split_pages} pages...")
        split_pdf(cfg.output_path, cfg.split_pages)


if __name__ == "__main__":
    main()