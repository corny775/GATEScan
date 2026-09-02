# GATEScan 🔍
### Paper Scanner & Answer Checker — OpenCV + Flask

> A three-stage intelligent examination evaluation system that turns smartphone photos of answer sheets into automatically graded results — no dedicated OMR hardware required.

---

<img width="1633" height="852" alt="Screenshot 2026-06-06 112134" src="https://github.com/user-attachments/assets/8f0925a1-8fe3-4b2e-877d-8c14059bd601" />



## Overview

**GATEScan** is a browser-based automated grading platform built for educational institutions that lack access to expensive Scantron-style hardware. Point a smartphone camera at an answer sheet, upload the photo, and the system handles the rest: perspective correction, MCQ bubble detection, and handwritten numerical answer recognition.

```
Raw Photo → Document Scan → OMR Grading → Numerical OCR → Final Score
```

---

## Features

- **📄 Document Scanner** — Detects answer-sheet boundaries from angled/distorted smartphone photos and applies perspective correction to produce a flat, clean scan
- **⭕ OMR Checker** — Segments bubble grids, counts filled pixels per cell, detects selected answers, and scores against a configurable answer key
- **🔢 Numerical Answer Checker** — Uses Tesseract OCR to recognise handwritten integers, decimals, and negative numbers with tolerance-based grading
- **🌐 Browser-Based Interface** — No Python or OpenCV installation needed on the client side; runs entirely in the browser via a Flask REST API
- **🎛️ Live Threshold Controls** — Adjust Canny edge thresholds in real time directly from the browser to handle varying lighting conditions
- **📊 Processing Stage Visualisation** — View 8 intermediate pipeline stages for each module to understand and debug the CV pipeline

---

## Demo

| Document Scanner | OMR Checker | Numerical Checker |
|:---:|:---:|:---:|
| Perspective correction from raw photo | Bubble detection + score overlay | OCR extraction + tolerance grading |

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python 3.x, Flask, Flask-CORS |
| **Computer Vision** | OpenCV (cv2), NumPy |
| **OCR** | Tesseract OCR (pytesseract) |
| **Frontend** | HTML5, CSS3, JavaScript (Fetch API) |
| **Data Transport** | Base64-encoded images over JSON REST API |

---

## How It Works

### Stage 1 — Document Scanner
1. Resize input image to 480×640 px
2. Grayscale conversion + Gaussian blur (5×5, σ=1)
3. Canny edge detection (user-adjustable thresholds)
4. Morphological dilation (2 iter) + erosion (1 iter)
5. Find external contours → select largest 4-point quadrilateral (area > 5000 px)
6. Reorder corner points (TL, TR, BL, BR)
7. Apply perspective warp (`getPerspectiveTransform`)
8. Adaptive thresholding + median blur → final clean scan

### Stage 2 — OMR Checker
1. Resize scan to 700×700 px
2. Detect rectangular contours → identify answer grid and grade region
3. Binary inverse thresholding at level 170
4. Split grid into 5×5 cells using `vsplit` / `hsplit`
5. Count non-zero pixels per cell → `argmax` per row = selected answer
6. Compare against answer key → compute `(correct / total) × 100`
7. Overlay green/red circles on graded sheet

### Stage 3 — Numerical Answer Checker
1. Detect answer-table contour (largest 4-point, not full page)
2. Perspective-warp table to fixed frame
3. Split into rows (one per question)
4. Per-row preprocessing: upscale → denoise → Otsu threshold → invert → morphological close
5. OCR with Tesseract; extract values using regex: `-?\d+\.?\d*`
6. Tolerance-based grading: `|detected − correct| ≤ tolerance`
7. Annotate rows with green/red indicators and correct values

---

## Project Structure

```
GATEScan/
├── app.py                  # Flask server + REST API endpoints
├── scanner_utils.py        # Document scanning helpers (contour, warp)
├── omr_utils.py            # OMR bubble segmentation + grading
├── numerical_utils.py      # OCR preprocessing + numerical evaluation
└── frontend/
    └── index.html          # Single-page browser interface
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/scan` | POST | Document scanning + perspective correction |
| `/save_scan` | POST | Save scanned image to uploads directory |
| `/check_omr` | POST | Evaluate MCQ/OMR answer sheet |
| `/check_numerical` | POST | Evaluate handwritten numerical answers |

---

## Installation

### Prerequisites
- Python 3.x
- Tesseract OCR installed on the system ([install guide](https://github.com/tesseract-ocr/tesseract))

### Setup

```bash
git clone https://github.com/yourusername/GATEScan.git
cd GATEScan
pip install opencv-python numpy flask flask-cors pytesseract
python app.py
```

Then open `frontend/index.html` in your browser (or navigate to `http://localhost:5000`).

---

## Usage

1. **Scan Tab** — Upload a smartphone photo of the answer sheet. Adjust Canny threshold sliders until the document boundary is clearly detected. Click **Scan** to generate a corrected scan.
2. **OMR Tab** — Upload (or send directly from the Scanner tab) the scanned sheet. Configure the number of questions and choices (e.g. 5 questions × 5 options). Enter the correct answer key. Click **Check Answers**.
3. **Numerical Tab** — Upload the scanned numerical section. Enter correct answers and an optional tolerance value. Click **Check Numerical Answers**.

---

## Limitations

- Contour detection requires moderate background contrast
- Severe shadows or motion blur may reduce accuracy
- OCR accuracy depends on handwriting clarity and digit spacing
- Heavily crumpled or partially occluded sheets may fail perspective correction
- Current OMR grid is fixed at 5×5 (configurable via frontend but not auto-detected)

---

## Future Work

- [ ] Deep learning–based OCR (CNN/Transformer) for improved handwriting recognition
- [ ] Direct in-browser camera capture via `MediaDevices` API
- [ ] Batch processing with CSV/Excel/PDF export
- [ ] Student ID / QR code integration for automatic result mapping
- [ ] Cloud deployment (Render, Railway, Heroku)
- [ ] Auto-detection of arbitrary answer-sheet grid layouts

---

## References

1. Bradski, G. (2000). *The OpenCV Library.* Dr. Dobb's Journal.
2. Suzuki, S. & Abe, K. (1985). Topological structural analysis of digitized binary images by border following. *CVGIP, 30*(1), 32–46.
3. Canny, J. (1986). A computational approach to edge detection. *IEEE TPAMI, 8*(6), 679–698.
4. Gonzalez, R. C. & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
5. Grinberg, M. (2018). *Flask Web Development* (2nd ed.). O'Reilly Media.

---

## License

MIT License — feel free to use and adapt for educational purposes.

---
