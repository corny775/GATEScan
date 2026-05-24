from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import scanner_utils
import omr_utils
import numerical_utils

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "../uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────
def decode_image(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    np_arr    = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_image(img):
    _, buffer = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")


# ══════════════════════════════════════════════════════════════════════
#  STAGE 1 — DOCUMENT SCANNER
# ══════════════════════════════════════════════════════════════════════
@app.route("/scan", methods=["POST"])
def scan():
    data       = request.get_json()
    img_b64    = data.get("image")
    threshold1 = int(data.get("threshold1", 200))
    threshold2 = int(data.get("threshold2", 200))

    img = decode_image(img_b64)
    heightImg, widthImg = 640, 480
    img      = cv2.resize(img, (widthImg, heightImg))
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)

    imgGray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur        = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold   = cv2.Canny(imgBlur, threshold1, threshold2)
    kernel         = np.ones((5, 5))
    imgDial        = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshFinal = cv2.erode(imgDial, kernel, iterations=1)

    imgContours   = img.copy()
    imgBigContour = img.copy()
    contours, _   = cv2.findContours(imgThreshFinal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    biggest, _ = scanner_utils.biggestContour(contours)
    result = {"found": False, "stages": {}}

    if biggest.size != 0:
        biggest       = scanner_utils.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        imgBigContour = scanner_utils.drawRectangle(imgBigContour, biggest, 2)

        pts1   = np.float32(biggest)
        pts2   = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored  = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        imgWarpColored  = imgWarpColored[20:imgWarpColored.shape[0]-20, 20:imgWarpColored.shape[1]-20]
        imgWarpColored  = cv2.resize(imgWarpColored, (widthImg, heightImg))
        imgWarpGray     = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        result["found"]         = True
        result["scanned_image"] = encode_image(imgWarpColored)
        result["stages"] = {
            "original":   encode_image(img),
            "gray":       encode_image(imgGray),
            "threshold":  encode_image(imgThreshFinal),
            "contours":   encode_image(imgContours),
            "bigContour": encode_image(imgBigContour),
            "warpColor":  encode_image(imgWarpColored),
            "warpGray":   encode_image(imgWarpGray),
            "adaptive":   encode_image(imgAdaptiveThre),
        }
    else:
        result["stages"] = {
            "original":   encode_image(img),
            "gray":       encode_image(imgGray),
            "threshold":  encode_image(imgThreshFinal),
            "contours":   encode_image(imgContours),
            "bigContour": encode_image(imgBlank),
            "warpColor":  encode_image(imgBlank),
            "warpGray":   encode_image(imgBlank),
            "adaptive":   encode_image(imgBlank),
        }
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════
#  SAVE SCANNED IMAGE
# ══════════════════════════════════════════════════════════════════════
@app.route("/save_scan", methods=["POST"])
def save_scan():
    data      = request.get_json()
    img_b64   = data.get("image")
    filename  = data.get("filename", "scanned_sheet.png")
    img       = decode_image(img_b64)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(save_path, img)
    return jsonify({"success": True, "path": save_path, "filename": filename})


# ══════════════════════════════════════════════════════════════════════
#  STAGE 2 — OMR CHECKER  (flexible questions & choices)
# ══════════════════════════════════════════════════════════════════════
@app.route("/check_omr", methods=["POST"])
def check_omr():
    data      = request.get_json()
    img_b64   = data.get("image")
    answers   = data.get("answers", [])       # correct answer indices (0-based)
    questions = int(data.get("questions", 5))
    choices   = int(data.get("choices", 5))

    img = decode_image(img_b64)
    widthImg = heightImg = 700
    img      = cv2.resize(img, (widthImg, heightImg))
    imgFinal = img.copy()
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)

    imgGray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur  = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 70)

    try:
        imgContours   = img.copy()
        imgBigContour = img.copy()
        contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

        rectCon       = omr_utils.rectContour(contours)
        biggestPoints = omr_utils.getCornerPoints(rectCon[0])
        gradePoints   = omr_utils.getCornerPoints(rectCon[1])

        if biggestPoints.size != 0 and gradePoints.size != 0:
            biggestPoints = omr_utils.reorder(biggestPoints)
            pts1   = np.float32(biggestPoints)
            pts2   = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            gradePoints = omr_utils.reorder(gradePoints)
            ptsG1   = np.float32(gradePoints)
            ptsG2   = np.float32([[0,0],[325,0],[0,150],[325,150]])
            matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
            imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))

            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh   = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

            boxes      = omr_utils.splitBoxes(imgThresh, questions, choices)
            myPixelVal = np.zeros((questions, choices))
            countR = countC = 0
            for image in boxes:
                myPixelVal[countR][countC] = cv2.countNonZero(image)
                countC += 1
                if countC == choices:
                    countC = 0
                    countR += 1

            myIndex = [int(np.where(myPixelVal[x] == np.amax(myPixelVal[x]))[0][0])
                       for x in range(questions)]

            grading = [1 if answers[x] == myIndex[x] else 0 for x in range(questions)]
            score   = (sum(grading) / questions) * 100

            omr_utils.showAnswers(imgWarpColored, myIndex, grading, answers, questions, choices)
            omr_utils.drawGrid(imgWarpColored, questions, choices)

            imgRawDrawings = np.zeros_like(imgWarpColored)
            omr_utils.showAnswers(imgRawDrawings, myIndex, grading, answers, questions, choices)
            invMatrix  = cv2.getPerspectiveTransform(pts2, pts1)
            imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))

            imgRawGrade = np.zeros_like(imgGradeDisplay, np.uint8)
            cv2.putText(imgRawGrade, str(int(score)) + "%", (70, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
            invMatrixG         = cv2.getPerspectiveTransform(ptsG2, ptsG1)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

            return jsonify({
                "success":          True,
                "score":            score,
                "grading":          grading,
                "selected_answers": myIndex,
                "correct_answers":  answers,
                "stages": {
                    "original":   encode_image(img),
                    "gray":       encode_image(imgGray),
                    "edges":      encode_image(imgCanny),
                    "contours":   encode_image(imgContours),
                    "bigContour": encode_image(imgBigContour),
                    "threshold":  encode_image(imgThresh),
                    "warped":     encode_image(imgWarpColored),
                    "final":      encode_image(imgFinal),
                }
            })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    return jsonify({"success": False, "error": "Could not detect answer sheet regions."})


# ══════════════════════════════════════════════════════════════════════
#  STAGE 3 — NUMERICAL ANSWER SHEET
#
#  FIX NOTES (vs previous broken version):
#  - Old code: used RETR_EXTERNAL and picked rectCon[0] which was always
#    the full-page border → warp produced a dark blank image → OCR got nothing.
#  - New code: delegates to numerical_utils.warpAnswerTable() which uses
#    RETR_LIST, filters out any contour covering >85% of the image area,
#    and selects the next largest rectangle (the answer table itself).
#    This warp correctly isolates just the numbered rows.
# ══════════════════════════════════════════════════════════════════════
@app.route("/check_numerical", methods=["POST"])
def check_numerical():
    data            = request.get_json()
    img_b64         = data.get("image")
    correct_answers = data.get("answers", [])   # list of strings e.g. ["-7.83", "15.26"]
    questions       = int(data.get("questions", 5))
    tolerance       = float(data.get("tolerance", 0))

    img = decode_image(img_b64)

    # Resize to consistent width while preserving aspect ratio
    TARGET_W       = 700
    h_orig, w_orig = img.shape[:2]
    scale_factor   = TARGET_W / w_orig
    img = cv2.resize(img, (TARGET_W, int(h_orig * scale_factor)))

    # Stage images for debugging
    imgGray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur  = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 30, 100)

    try:
        # Draw all contours for the debug view
        imgContours  = img.copy()
        all_cnts, _  = cv2.findContours(imgCanny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgContours, all_cnts, -1, (0, 255, 0), 2)

        # ── Warp only the answer table, not the whole page ────────────
        # warpAnswerTable() skips the full-page border contour (>85% area)
        # and picks the next largest rectangle (the numbered answer table).
        # It returns a clean warped crop + a debug image showing the detected
        # table border in red.
        imgWarp, imgContourDebug = numerical_utils.warpAnswerTable(img, questions)

        # ── Split the warped table into one row per question ──────────
        row_imgs = numerical_utils.splitRows(imgWarp, questions)

        # ── OCR each row ──────────────────────────────────────────────
        detected      = []
        row_annotated = []
        for row in row_imgs:
            text = numerical_utils.extractNumber(row)
            detected.append(text)
            row_annotated.append(encode_image(row))

        # ── Grade every question ──────────────────────────────────────
        grading = [
            numerical_utils.gradeAnswer(
                detected[i],
                str(correct_answers[i]) if i < len(correct_answers) else "",
                tolerance
            )
            for i in range(questions)
        ]
        score = (sum(grading) / questions) * 100

        # ── Annotate the warped table with results ────────────────────
        imgFinal = numerical_utils.drawNumericalResults(
            imgWarp.copy(), detected, grading, correct_answers, questions
        )

        return jsonify({
            "success":         True,
            "score":           score,
            "grading":         grading,
            "detected":        detected,
            "correct_answers": correct_answers,
            "row_images":      row_annotated,
            "stages": {
                "original":  encode_image(img),
                "gray":      encode_image(imgGray),
                "edges":     encode_image(imgCanny),
                "contours":  encode_image(imgContourDebug),  # red outline = detected table
                "warped":    encode_image(imgWarp),
                "final":     encode_image(imgFinal),
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)