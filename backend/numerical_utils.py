import cv2
import numpy as np
import re
import pytesseract

# On Windows, set the path to your Tesseract installation:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Tesseract config tuned for numbers (digits, minus, dot) ──────────
TESS_CONFIG = (
    "--oem 3 --psm 7 "
    "-c tessedit_char_whitelist=0123456789.-"
)


def findAnswerTableContour(contours, img_w, img_h):
    """
    From all detected rectangular contours, find the one most likely to be
    the answer table — i.e. NOT the full page boundary, and reasonably tall.

    Strategy:
      - Sort all 4-point contours by area descending.
      - Skip any contour whose bounding box covers > 85% of the image area
        (that's the full-page border, not the answer table).
      - Return the largest remaining contour.
    """
    img_area   = img_w * img_h
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:               # ignore tiny noise contours
            continue
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:         # only rectangles
            continue
        # Reject if it covers almost the whole image (full-page border)
        if area / img_area > 0.85:
            continue
        candidates.append((area, cnt))

    if not candidates:
        return None

    # Sort by area descending — biggest non-page rectangle is the answer table
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def reorderPoints(pts):
    """
    Reorder 4 corner points into [top-left, top-right, bottom-left, bottom-right].
    Identical to the reorder() in omr_utils / scanner_utils.
    """
    pts = pts.reshape((4, 2))
    new_pts = np.zeros((4, 1, 2), dtype=np.int32)
    add  = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    new_pts[0] = pts[np.argmin(add)]    # top-left     (smallest x+y)
    new_pts[3] = pts[np.argmax(add)]    # bottom-right (largest  x+y)
    new_pts[1] = pts[np.argmin(diff)]   # top-right    (smallest x-y)
    new_pts[2] = pts[np.argmax(diff)]   # bottom-left  (largest  x-y)
    return new_pts


def warpAnswerTable(img, questions):
    """
    Detect the answer table rectangle and warp it to a flat 500 × (questions * 80) px image.
    Returns (warped_img, debug_contour_img) or raises RuntimeError if table not found.
    """
    TARGET_W = 500
    TARGET_H = questions * 80      # 80 px per row gives enough height for OCR

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blur, 30, 100)

    # Dilate to close gaps in table lines
    kernel = np.ones((3, 3), np.uint8)
    edges  = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    debug_img = img.copy()
    cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)

    h, w   = img.shape[:2]
    table_cnt = findAnswerTableContour(contours, w, h)

    if table_cnt is None:
        raise RuntimeError(
            "Could not locate the answer table. "
            "Make sure the table has a visible border and the image is well-lit."
        )

    # Get and reorder corner points
    peri   = cv2.arcLength(table_cnt, True)
    approx = cv2.approxPolyDP(table_cnt, 0.02 * peri, True)

    if len(approx) != 4:
        raise RuntimeError("Answer table contour does not have exactly 4 corners.")

    corners = reorderPoints(approx)
    cv2.drawContours(debug_img, [corners], -1, (0, 0, 255), 4)

    # Perspective warp
    pts1   = np.float32(corners)
    pts2   = np.float32([
        [0,        0       ],
        [TARGET_W, 0       ],
        [0,        TARGET_H],
        [TARGET_W, TARGET_H]
    ])
    matrix  = cv2.getPerspectiveTransform(pts1, pts2)
    warped  = cv2.warpPerspective(img, matrix, (TARGET_W, TARGET_H))

    return warped, debug_img


def preprocessRow(row_img):
    """
    Convert a single row crop to a clean binary image for Tesseract OCR.

    Steps:
      1. Grayscale
      2. Upscale 3× (Tesseract works much better on larger images)
      3. Gaussian denoise
      4. Otsu threshold (automatically handles light/dark backgrounds)
      5. Invert if background is dark (Tesseract needs dark text on light bg)
      6. Morphological close to reconnect broken digit strokes
    """
    gray  = cv2.cvtColor(row_img, cv2.COLOR_BGR2GRAY)

    # Upscale for better OCR accuracy
    scale = 3
    h, w  = gray.shape
    gray  = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Denoise
    gray  = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu binarisation
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tesseract needs dark text on a light background
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # Close small gaps in digit strokes
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def splitRows(warped_img, questions):
    """
    Split the warped answer-table image into one horizontal strip per question.

    The warped image is assumed to contain ONLY the answer table (no title,
    logo, or question-number column to the left of the number column).
    We still skip the leftmost 25% of each row in case the question number
    bleeds into the crop.

    Returns a list of BGR row images, one per question.
    """
    h, w  = warped_img.shape[:2]
    row_h = h // questions
    rows  = []

    for i in range(questions):
        y1  = i * row_h
        y2  = min(y1 + row_h, h)
        row = warped_img[y1:y2, 0:w]

        # Skip left 25% to remove question-number column
        answer_start = int(w * 0.25)
        row = row[:, answer_start:]

        # Trim 8% top and bottom to avoid row-border lines
        pad = max(3, int(row_h * 0.08))
        row = row[pad : row_h - pad, :]

        rows.append(row)

    return rows


def extractNumber(row_img):
    """
    Run Tesseract OCR on a row crop and extract a valid numerical string.

    Handles:
      - Integers:  56,  9999,  0
      - Decimals:  0.4,  3.681,  0.0047
      - Negatives: -39,  -7.83,  -25.14

    Returns the first matched number string, or '' if nothing found.
    """
    processed = preprocessRow(row_img)
    raw       = pytesseract.image_to_string(processed, config=TESS_CONFIG).strip()

    # Extract the first valid number pattern from the OCR output
    tokens = re.findall(r"-?\d+\.?\d*", raw)
    if tokens:
        return tokens[0]
    return ""


def gradeAnswer(detected_str, correct_str, tolerance=0):
    """
    Compare a detected answer string to the correct answer string numerically.

    - Both values are cast to float so that "3.0" == "3" and "-7.83" == "-7.83".
    - tolerance: maximum allowed absolute difference (default 0 = exact match).
      Useful for floating-point answers, e.g. tolerance=0.01 accepts ±0.01.

    Returns 1 if correct, 0 if wrong or unparseable.
    """
    try:
        detected = float(detected_str)
        correct  = float(correct_str)
        return 1 if abs(detected - correct) <= tolerance else 0
    except (ValueError, TypeError):
        return 0


def drawNumericalResults(img, detected, grading, correct_answers, questions):
    """
    Annotate the warped answer-table image with grading results.

    For each row:
      - Left side:  green "OK" or red "X"
      - Right side: "Got: <detected>"  in green (correct) or red (wrong)
      - Wrong rows also show "Ans: <correct>" in green below

    Top banner shows the overall score.
    A thin grey line separates each row.
    """
    h, w     = img.shape[:2]
    row_h    = h // questions
    font     = cv2.FONT_HERSHEY_SIMPLEX

    # The answer-number column starts 25% in (matches splitRows offset)
    ans_col  = int(w * 0.25)

    for i in range(questions):
        cy  = i * row_h + row_h // 2      # vertical centre of this row
        det = detected[i]       if i < len(detected)         else ""
        cor = str(correct_answers[i]) if i < len(correct_answers) else ""
        ok  = grading[i] == 1

        green = (0, 200, 80)
        red   = (0, 60, 220)
        color = green if ok else red

        # ── Tick / cross ──────────────────────────────────────────────
        cv2.putText(img, "OK" if ok else "X",
                    (8, cy + 8), font, 0.65, color, 2, cv2.LINE_AA)

        # ── Detected value ────────────────────────────────────────────
        det_label = f"Got: {det}" if det else "Got: ?"
        cv2.putText(img, det_label,
                    (ans_col + 10, cy - 4), font, 0.60, color, 2, cv2.LINE_AA)

        # ── Correct value (wrong answers only) ───────────────────────
        if not ok:
            cv2.putText(img, f"Ans: {cor}",
                        (ans_col + 10, cy + 18), font, 0.50, green, 1, cv2.LINE_AA)

        # ── Row separator line ────────────────────────────────────────
        cv2.line(img,
                 (0,   (i + 1) * row_h),
                 (w,   (i + 1) * row_h),
                 (180, 180, 180), 1)

    # ── Score banner at the very top ─────────────────────────────────
    total     = questions if questions > 0 else 1
    score_pct = int((sum(grading) / total) * 100)
    cv2.rectangle(img, (0, 0), (w, 34), (20, 20, 20), cv2.FILLED)
    cv2.putText(img,
                f"Score: {score_pct}%  ({sum(grading)}/{questions} correct)",
                (10, 23), font, 0.65, (0, 220, 120), 2, cv2.LINE_AA)

    return img