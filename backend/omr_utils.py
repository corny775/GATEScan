import cv2
import numpy as np

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def rectContour(contours):
    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return approx

def splitBoxes(img, questions=5, choices=5):
    """Split the warped sheet into a grid of (questions x choices) boxes."""
    rows = np.vsplit(img, questions)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, choices)
        for box in cols:
            boxes.append(box)
    return boxes

def drawGrid(img, questions=5, choices=5):
    secW = int(img.shape[1] / choices)
    secH = int(img.shape[0] / questions)
    for i in range(max(questions, choices) + 1):
        if i <= questions:
            pt1 = (0, secH * i)
            pt2 = (img.shape[1], secH * i)
            cv2.line(img, pt1, pt2, (255, 255, 0), 2)
        if i <= choices:
            pt3 = (secW * i, 0)
            pt4 = (secW * i, img.shape[0])
            cv2.line(img, pt3, pt4, (255, 255, 0), 2)
    return img

def showAnswers(img, myIndex, grading, ans, questions=5, choices=5):
    secW = int(img.shape[1] / choices)
    secH = int(img.shape[0] / questions)
    for x in range(questions):
        myAns = myIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2
        radius = min(secW, secH) // 3
        if grading[x] == 1:
            cv2.circle(img, (cX, cY), radius, (0, 255, 0), cv2.FILLED)
        else:
            cv2.circle(img, (cX, cY), radius, (0, 0, 255), cv2.FILLED)
            correctAns = ans[x]
            cx2 = (correctAns * secW) + secW // 2
            cv2.circle(img, (cx2, cY), radius // 2, (0, 255, 0), cv2.FILLED)