import cv2
import numpy as np
from pprint import pprint

TARGET_AREA = 100000

def setLabel(image, s, contour):
    (text_width, text_height), baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 3, 1)
    x, y, width, height = cv2.boundingRect(contour)
    print(x, y, width, height)
    pt_x = x + int((width - text_width) / 2)
    pt_y = y + int((height + text_height) / 2)
    cv2.rectangle(image, (pt_x, pt_y + baseline), (pt_x + text_width, pt_y-text_height), (200, 200, 200), cv2.FILLED)
    cv2.putText(image, s, (pt_x, pt_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 1, 8)

def findMaxArea(contours):
    max_contour = None
    max_area = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        _, _, w, h = cv2.boundingRect(cnt)

        if (w * h) * 0.4 > area:
            continue

        if w > h:
            continue

        if area > max_area:
            max_area = area
            max_contour = cnt

    if max_area < TARGET_AREA:
        max_area = -1

    return max_area, max_contour


def toPerspectiveImage(img, points):
    print(points.shape)
    points = points.reshape((-1, 2))
    # 항상 4개의 점이 검출되지 않음.. breadboard4.jpg
    sm = points.sum(axis = 1)
    diff = np.diff(points, axis=1)

    topLeft = points[np.argmin(sm)]
    bottomRight = points[np.argmax(sm)]
    topRight = points[np.argmin(diff)]
    bottomLeft = points[np.argmax(diff)]

    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])

    width = max([w1, w2])
    height = max([h1, h2])

    pts2 = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])

    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    return pts2, cv2.warpPerspective(img, mtrx, (width, height))

source = cv2.imread('breadboard3.jpg')
source_copy = source.copy()

source_gray = cv2.GaussianBlur(source, (5, 5), cv2.BORDER_DEFAULT)
source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
ret, otsu = cv2.threshold(source_gray, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

_, cnt = findMaxArea(contours)
epsilon = 0.02 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

size = len(approx)
cv2.line(source_copy, tuple(approx[0][0]), tuple(approx[size-1][0]), (0, 255, 0), 3)
for k in range(size-1):
    cv2.line(source_copy, tuple(approx[k][0]), tuple(approx[k+1][0]), (0, 255, 0), 3)

cnt, source_copy = toPerspectiveImage(source_copy, approx)
new_target = source_copy.copy()
draw_target = source_copy.copy()

new_target = cv2.GaussianBlur(new_target, (5, 5), cv2.BORDER_DEFAULT)
new_target = cv2.cvtColor(new_target, cv2.COLOR_BGR2GRAY)
ret, new_target = cv2.threshold(new_target, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(new_target, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

z = np.zeros(new_target.shape)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 100 and area < 500:
        pin = cnt.reshape(-1, 2)
        M = cv2.moments(pin, False)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        cv2.circle(z, (cX, cY), 5, (255, 255, 0), -1)

cv2.imshow('BreadBoard Pin Area', z)

cv2.waitKey(0)
cv2.destroyAllWindows()