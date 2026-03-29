"""
capture_to_pdf.py — Automatic capture of pages from a VirtualBox window and save to PDF.

Usage:
    python capture_to_pdf.py [output.pdf] [--debug] [--delay 1.3] [--max-pages 999]
                             [--icon arrow.png] [--quality 50]
                             [--trim-top 140] [--trim-bottom 90] [--trim-left 160] [--trim-right 100]
                             [--split N]

  --split N   After saving, splits the PDF into chunks of N pages.
              Generated files use the output base name with suffix _NNNN_MMMM.pdf
              (e.g. book_0001_0200.pdf, book_0201_0400.pdf, book_0401_end.pdf).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass, field
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
    Most fields map 1-to-1 to a CLI argument (see parse_args).
    """
    output_path: str = "output.pdf"
    icon_file: str = "arrow.png"

    # Pixels to crop from each edge of the raw window screenshot.
    # This removes window chrome (title bar, borders, status bar) so that
    # only the document content is saved.
    trim_top: int = 140
    trim_bottom: int = 90
    trim_left: int = 160
    trim_right: int = 100

    # Minimum confidence score (0-1) for template matching to accept a hit.
    # Lower values tolerate more visual variation but risk false positives.
    match_threshold: float = 0.75

    # Minimum number of orange/yellow HSV pixels to even consider that an
    # orange selection box might be present.
    orange_pixel_min: int = 200

    # A detected line must be longer than this (pixels) to count as a
    # "real" edge of the selection rectangle (filters out short artifacts).
    line_length_min: int = 80

    # How many long lines must be found before we declare an orange box present.
    long_lines_min: int = 2

    # Seconds to wait between clicking the "next page" icon and capturing
    # the next page. Increase if the VM is slow to animate page transitions.
    delay: float = 1.3

    # Safety cap: stop after this many pages even if the end is not detected.
    max_pages: int = 2000

    # Pillow JPEG quality for each page (1 = smallest file, 95 = best quality).
    jpeg_quality: int = 50

    # When True, intermediate images (HSV, mask, trimmed page) are written to
    # disk with a timestamp/page-number suffix so you can inspect what the
    # script is "seeing" at each step.
    debug: bool = False

    # If > 0, the final PDF is split into chunks of this many pages.
    # 0 means no splitting.
    split_pages: int = 0


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------

def get_virtualbox_window(title_hint: str = "VirtualBox", title_filter: str = "mlol"):
    """
    Find and return the target VirtualBox window.

    First, all windows whose title contains `title_hint` are collected.
    Then, if `title_filter` is set, the list is narrowed to windows whose
    title also contains that string (case-insensitive). This lets us pick
    the correct VM when multiple VirtualBox windows are open (e.g. the
    Manager vs a running VM named "mlol").

    Raises RuntimeError if no VirtualBox window is found at all.
    """
    wins = [w for w in pwc.getAllWindows() if title_hint in w.title]
    if not wins:
        raise RuntimeError(f"No window with '{title_hint}' in the title found.")
    if title_filter:
        filtered = [w for w in wins if title_filter.lower() in w.title.lower()]
        if filtered:
            wins = filtered
        else:
            # Fall back gracefully instead of crashing.
            print(f"Warning: no window with '{title_filter}' in the title, "
                  f"using the first available: '{wins[0].title}'")
    return wins[0]


def screenshot_window(win) -> tuple[np.ndarray, int, int]:
    """
    Capture the full contents of `win` using mss (a fast screen-capture lib).

    Returns:
        image   - HxWx3 uint8 array in RGB colour order.
        win_x   - absolute X coordinate of the window's top-left corner on
                  the physical display (needed later to convert relative icon
                  coordinates into absolute click coordinates).
        win_y   - absolute Y coordinate of the window's top-left corner.
    """
    with mss() as sct:
        raw = sct.grab({
            "left": win.left,
            "top": win.top,
            "width": win.width,
            "height": win.height,
        })
        # mss returns raw RGB bytes; reshape them into a proper 2-D image array.
        img_np = np.frombuffer(raw.rgb, dtype=np.uint8).reshape((raw.height, raw.width, 3))
    # .copy() detaches the array from the mss internal buffer which is freed
    # when the `with` block exits.
    return img_np.copy(), win.left, win.top


def trim_screenshot(screenshot: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Remove the window chrome (title bar, scroll bars, status bar, borders)
    from a raw screenshot by slicing away a fixed number of pixels on each side.

    This is intentionally done only ONCE per page capture cycle. The trimmed
    image is then reused for both the orange-box check and the PDF export,
    avoiding redundant work and ensuring consistency.
    """
    h, w, _ = screenshot.shape
    return screenshot[
        cfg.trim_top : h - cfg.trim_bottom,
        cfg.trim_left : w - cfg.trim_right,
    ]


# ---------------------------------------------------------------------------
# Orange box detection
# ---------------------------------------------------------------------------

def has_orange_box(image_np: np.ndarray, cfg: Config, debug_suffix: str = "") -> bool:
    """
    Detect whether the VirtualBox UI is showing an orange selection highlight,
    which signals that the page transition animation is still in progress.

    Detection strategy:
      1. Convert the image to HSV colour space. Orange (#FFA500) maps to a
         narrow yellow-ish hue band in HSV (approximately H=18-25).
      2. Threshold to a binary mask of "orange-ish" pixels.
      3. If too few such pixels exist, the box is absent → return False early.
      4. Run Canny edge detection and Hough line transform on the mask.
         A real selection rectangle produces at least `long_lines_min` long
         straight edges; random noise does not.

    Returns True only when both the pixel count and line count thresholds
    are met, minimising false positives from UI elements that happen to be
    orange-coloured.
    """
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # HSV range for the orange/yellow highlight colour.
    lower = np.array([18, 200, 200])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    if cfg.debug and debug_suffix:
        cv2.imwrite(f"debug_hsv_{debug_suffix}.png", hsv)
        cv2.imwrite(f"debug_mask_{debug_suffix}.png", mask)

    yellow_pixels = int(np.sum(mask > 0))
    if yellow_pixels < cfg.orange_pixel_min:
        # Not enough coloured pixels — box is definitely absent.
        return False

    # Look for straight edges that would form the rectangle sides.
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


def wait_until_clean(win, cfg: Config, page_num: int):
    """
    Poll the window repeatedly until the orange selection box disappears.

    The box appears briefly whenever the user (or script) clicks the "next
    page" arrow. We must wait for it to clear before capturing, otherwise
    the saved page image will contain the orange overlay.

    The trimmed image is computed here (once) and returned alongside the
    original screenshot and window coordinates, so callers don't need to
    trim again.

    Returns:
        screenshot  - full, untrimmed window image (used for icon search).
        trimmed     - cropped document area (used for hashing and PDF export).
        win_x, win_y - window origin on the physical display.
    """
    while True:
        screenshot, win_x, win_y = screenshot_window(win)
        trimmed = trim_screenshot(screenshot, cfg)

        if not has_orange_box(trimmed, cfg, debug_suffix=str(page_num) if cfg.debug else ""):
            return screenshot, trimmed, win_x, win_y

        print("  Orange box detected, waiting...")
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# Template matching (icon)
# ---------------------------------------------------------------------------

def load_icon(icon_file: str) -> np.ndarray:
    """
    Load the "next page" arrow icon from disk.
    Called once at startup so the file I/O and decode cost is paid only once,
    not once per page.
    """
    icon = cv2.imread(icon_file)
    if icon is None:
        raise RuntimeError(f"Unable to load icon: {icon_file}")
    return icon


def find_icon_in_image(screenshot: np.ndarray, icon: np.ndarray, threshold: float):
    """
    Locate the arrow icon inside the full window screenshot using normalised
    cross-correlation template matching (TM_CCOEFF_NORMED).

    The search runs on the untrimmed screenshot because the icon lives in the
    window chrome area that is cropped out of the content image.

    Returns:
        (top_left, icon_width, icon_height) if the match score >= threshold.
        None if the icon is not found or the match is too weak.
    """
    res = cv2.matchTemplate(screenshot, icon, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val < threshold:
        return None

    return max_loc, icon.shape[1], icon.shape[0]


# ---------------------------------------------------------------------------
# Page hashing
# ---------------------------------------------------------------------------

def image_md5(image_np: np.ndarray, size: int = 128) -> str:
    """
    Compute a fast perceptual fingerprint of a page image.

    The image is downscaled to `size`×`size` and converted to grayscale
    before hashing. This makes the comparison robust to minor rendering
    differences while still reliably detecting when the document has looped
    back to a page already seen (which signals the end of the document).

    128px gives a good balance between collision resistance and speed.
    64px was used previously but is more prone to false matches on pages
    with sparse content.
    """
    small = cv2.resize(image_np, (size, size))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    return hashlib.md5(gray.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# In-memory image accumulation
# ---------------------------------------------------------------------------

def compress_page(image_np: np.ndarray, quality: int) -> bytes:
    """
    Convert a page image to a compressed JPEG and return it as raw bytes.

    Grayscale conversion halves the data compared to RGB. Storing bytes
    (rather than keeping a PIL Image open on a BytesIO buffer) means the
    garbage collector can safely reclaim the buffer immediately, avoiding
    subtle use-after-free bugs when many pages accumulate.
    """
    img = Image.fromarray(image_np).convert("L")  # L = 8-bit grayscale
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Page capture
# ---------------------------------------------------------------------------

def process_page(
    icon: np.ndarray,
    cfg: Config,
    prev_md5: str | None,
    page_num: int,
) -> tuple[bool, str | None, bytes | None]:
    """
    Capture one page of the document and advance to the next.

    Steps:
      1. Wait for the UI to settle (no orange box).
      2. Trim the screenshot once and reuse the result.
      3. Hash the trimmed image and compare with the previous page.
         If they match, the document has looped → stop.
      4. Compress the page to JPEG bytes.
      5. Find the "next" arrow icon and click it.

    Returns:
        success     - False when capture should stop (duplicate page or
                      icon not found), True to keep going.
        current_md5 - hash of this page (passed as prev_md5 on the next call).
        jpeg_bytes  - compressed page image, or None if this was a duplicate.
    """
    win = get_virtualbox_window()
    # wait_until_clean performs the trim internally and returns both the full
    # screenshot (for icon search) and the trimmed image (for everything else).
    screenshot, trimmed, win_x, win_y = wait_until_clean(win, cfg, page_num)

    if cfg.debug:
        cv2.imwrite(f"debug_page_{page_num}.png", cv2.cvtColor(trimmed, cv2.COLOR_RGB2BGR))

    current_md5 = image_md5(trimmed)

    # End-of-document check: must happen BEFORE compressing to avoid saving
    # a duplicate last page into the PDF.
    if prev_md5 and current_md5 == prev_md5:
        print("  Page identical to previous — end of document.")
        return False, prev_md5, None

    jpeg_bytes = compress_page(trimmed, cfg.jpeg_quality)

    # The icon is searched in the FULL (untrimmed) screenshot because it sits
    # in the window chrome, outside the trimmed document area.
    found = find_icon_in_image(screenshot, icon, cfg.match_threshold)
    if not found:
        print("  Icon not found in screenshot.")
        # Return the bytes anyway so the last visible page is still saved.
        return False, current_md5, jpeg_bytes

    # Convert the icon's position relative to the window into absolute screen
    # coordinates so pyautogui can click the right spot.
    (px, py), iw, ih = found
    click_x = win_x + px + iw // 2
    click_y = win_y + py + ih // 2

    print(f"  Click at ({click_x}, {click_y})")
    pyautogui.click(click_x, click_y)

    return True, current_md5, jpeg_bytes


# ---------------------------------------------------------------------------
# PDF splitting
# ---------------------------------------------------------------------------

def split_pdf(output_path: str, step: int) -> None:
    """
    Split a PDF into equal-sized chunks using pypdf (pure Python, no system tools).

    Chunk naming convention (1-based page numbers):
      <base>_0001_0200.pdf   - full chunk
      <base>_0201_end.pdf    - final partial chunk

    The original PDF is left untouched.
    """
    base = Path(output_path).stem
    out_dir = Path(output_path).parent

    reader = PdfReader(output_path)
    total = len(reader.pages)
    print(f"  Total pages in PDF: {total}")

    start = 0             # 0-based index into reader.pages
    block_start_label = 1 # 1-based label used in file names

    while start < total:
        end = min(start + step, total)  # exclusive upper bound for slicing
        end_label = start + step        # what the end page *would* be if the chunk were full
        is_last = end < start + step    # True when there aren't enough pages to fill the chunk

        # Copy the selected page range into a fresh writer.
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        if is_last:
            out_file = out_dir / f"{base}_{block_start_label:04d}_end.pdf"
        else:
            out_file = out_dir / f"{base}_{block_start_label:04d}_{end_label:04d}.pdf"

        print(f"  Pages {block_start_label}-{end} → {out_file.name}")
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
    Assemble the list of per-page JPEG byte strings into a single PDF file
    using Pillow's multi-image save feature.

    All pages are opened from their in-memory byte buffers so no temporary
    files are written to disk during the process.
    """
    if not pages_bytes:
        print("No images to save.")
        return

    images = [Image.open(BytesIO(b)) for b in pages_bytes]
    # Pillow writes all images as pages when save_all=True and
    # append_images contains the remaining frames.
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
    )
    size_kb = Path(output_path).stat().st_size // 1024
    print(f"PDF saved: {output_path}  ({size_kb} KB, {len(images)} pages)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    """
    Parse command-line arguments and return a populated Config instance.
    All parameters have sensible defaults so the script can be run with
    just the output filename (or even no arguments at all).
    """
    parser = argparse.ArgumentParser(
        description="Capture pages from VirtualBox and save them as PDF."
    )
    parser.add_argument("output", nargs="?", default="output.pdf",
                        help="Output PDF file (default: output.pdf)")
    parser.add_argument("--icon", default="arrow.png",
                        help="Template image of the 'next page' icon to click")
    parser.add_argument("--delay", type=float, default=1.3,
                        help="Seconds to wait after clicking before capturing the next page")
    parser.add_argument("--max-pages", type=int, default=999,
                        help="Stop after this many pages even if the end is not detected")
    parser.add_argument("--quality", type=int, default=50,
                        help="JPEG quality per page: 1 (smallest) - 95 (best), default 50")
    parser.add_argument("--trim-top",    type=int, default=140)
    parser.add_argument("--trim-bottom", type=int, default=90)
    parser.add_argument("--trim-left",   type=int, default=160)
    parser.add_argument("--trim-right",  type=int, default=100)
    parser.add_argument("--debug", action="store_true",
                        help="Write intermediate images to disk for troubleshooting")
    parser.add_argument("--split", type=int, default=0, metavar="N",
                        help="Split the output PDF into chunks of N pages (0 = disabled)")

    args = parser.parse_args()
    return Config(
        output_path=args.output,
        icon_file=args.icon,
        delay=args.delay,
        max_pages=args.max_pages,
        jpeg_quality=args.quality,
        trim_top=args.trim_top,
        trim_bottom=args.trim_bottom,
        trim_left=args.trim_left,
        trim_right=args.trim_right,
        debug=args.debug,
        split_pages=args.split,
    )


def main() -> None:
    """
    Main loop: capture pages one by one until the end of the document is
    reached (duplicate page hash, icon not found, or max-pages limit hit),
    then assemble and optionally split the PDF.
    """
    cfg = parse_args()

    # Load the icon template once; find_icon_in_image reuses it every page.
    icon = load_icon(cfg.icon_file)

    pages: list[bytes] = []  # accumulates compressed JPEG bytes for each page
    prev_md5: str | None = None
    page_num = 0

    print("Starting capture...")

    while page_num < cfg.max_pages:
        page_num += 1
        print(f"Page {page_num}...")

        success, prev_md5, jpeg_bytes = process_page(icon, cfg, prev_md5, page_num)

        if jpeg_bytes is not None:
            pages.append(jpeg_bytes)

        if not success:
            break  # end of document or unrecoverable error

        # Give the VM time to render the next page before the next capture.
        time.sleep(cfg.delay)

    print(f"Capture completed. Total pages: {len(pages)}")
    save_pdf(pages, cfg.output_path)

    if cfg.split_pages > 0:
        print(f"\nSplitting into chunks of {cfg.split_pages} pages...")
        split_pdf(cfg.output_path, cfg.split_pages)


if __name__ == "__main__":
    main()
