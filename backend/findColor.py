import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

SCALE = 0.75

''' 
    브레드보드 사이즈 X; 5.5cm
    브레드보드 사이즈 Y; 8.5cm
    센터포인트 (cX, cY)에서
    1cm = 37.795275591px

    X -> 207.87...
    Y -> 321.26...
    
    6등분으로 나눌 수 있네

    0.5 * 0.3333...

    핀쪽은 1.8315cm
    전압핀쪽은 0.91575cm

    결국 뒷배경이 깔끔하게 따여야 비율적인 계산이 가능함.
'''

BREADBOARD_AREA = {
    "left_area": (249, 71, 540, 1721),
    "left_voltage_area": (38, 98, 162, 1695),
    "right_area": (631, 75, 920, 1720),
    "right_voltage_area": (1010, 88, 1131, 1704),
}

def getClearBreadboardMap():
    breadboard_map = cv2.imread("./images/Breadboard/BreadboardMap.jpg")

    y, x, _ = breadboard_map.shape
    new_line = int(0.66 * x/2)

    area = {
        '1': (0, int(breadboard_map.shape[1] / 2 - new_line - 1)),
        '2': (int(breadboard_map.shape[1] / 2 - new_line), int(breadboard_map.shape[1] / 2 - 1)),
        '3': (int(breadboard_map.shape[1] / 2), int(breadboard_map.shape[1] / 2 + new_line - 1)),
        '4': (int(breadboard_map.shape[1] / 2 + new_line), breadboard_map.shape[1])
    }

    pins = getBreadboardPin(breadboard_map, area)
    pins['shape'] = breadboard_map.shape

    print(pins)

    with open('./static/data/pinmap.json', "w") as f:
        json.dump(pins, f)


def getSectionPointMap(target_img):
    pts = []
    img_copy = target_img.copy()
    # cv2.imshow('process_before', img_copy)

    process_image = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    process_image = cv2.medianBlur(process_image, ksize=7)
    # process_image = cv2.adaptiveThreshold(process_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 15)
    _, process_image = cv2.threshold(process_image, -1, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ada = process_image.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    process_image = cv2.erode(process_image, kernel, anchor=(kernel.shape[0]-1, -1), iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    process_image = cv2.dilate(process_image, kernel, anchor=(-1, -1), iterations=3)

    dil = process_image.copy()

    contours, _ = cv2.findContours(process_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow('res', np.concatenate((ada, dil, process_image), axis=1))

    for contour in contours:
        area = cv2.contourArea(contour)

        # Pin 영역 크기 정도의 것들만 추림
        if area > 100:
            pin = contour.reshape(-1, 2)
            M = cv2.moments(pin, False)
            pts.append(
                (int(M['m10'] / M['m00']),  
                int(M['m01'] / M['m00']))
            )
            cv2.circle(img_copy, pts[-1], 5, (255, 255, 0), -1)
    return pts, img_copy

def getBreadboardPin(src: np.ndarray, area=None):
    result = src.copy()
    breadboard_pinmap = {}

    right_pin_area = result[
        :, area['2'][0]:area['2'][1], :]

    right_voltage_area = result[
        :, area['1'][0]:area['1'][1], :]

    left_pin_area = result[
        :, area['3'][0]:area['3'][1], :]

    left_voltage_area = result[
        :, area['4'][0]:area['4'][1], :]

    # 오른쪽 핀영역에 대한 좌표맵 찾기
    pts, right_pin_area = getSectionPointMap(right_pin_area)

    breadboard_pinmap['2'] = {
        'start': area['2'][0],
        'points': pts
    }

    # 오른쪽 전압 핀영역에 대한 좌표맵 찾기
    pts, right_voltage_area = getSectionPointMap(right_voltage_area)

    breadboard_pinmap['1'] = {
        'start': area['1'][0],
        'points': pts
    }

    # 왼쪽 핀영역에 대한 좌표맵 찾기
    pts, left_pin_area = getSectionPointMap(left_pin_area)

    breadboard_pinmap['3'] = {
        'start': area['3'][0],
        'points': pts
    }

    # 왼쪽 전압 핀영역에 대한 좌표맵 찾기
    pts, left_voltage_area = getSectionPointMap(left_voltage_area)

    breadboard_pinmap['4'] = {
        'start': area['4'][0],
        'points': pts
    }

    return breadboard_pinmap


def grab_cut(s: np.ndarray, ROI):
    src = s.copy()
    mask_img = np.zeros(src.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(src, mask_img, ROI, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
    mask_img = np.where((mask_img == 2) | (mask_img == 0), 0, 1).astype('uint8')
    img = src * mask_img[:, :, np.newaxis]

    # cv2.imshow('grab_cunt', img)
    return img

def toPerspectiveImage(img, points):
    print(points.shape)
    print(points)

    if points.ndim != 2:
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

def toBinary(img):
    img_copy = img.copy()
    process_image = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    process_image = cv2.medianBlur(process_image, ksize=7)
    process_image = cv2.adaptiveThreshold(process_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 15)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    process_image = cv2.erode(process_image, kernel, anchor=(kernel.shape[0]-1, -1), iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    process_image = cv2.dilate(process_image, kernel, anchor=(-1, -1), iterations=3)
    
    return process_image

def fourier_test():
    breadboard_map = cv2.imread("images/breadboard_F.jpg", 0)
    # cv2.imshow("main", breadboard_map)

    f = np.fft.fft2(breadboard_map)
    fshift = np.fft.fftshift(f)

    rows, cols = breadboard_map.shape
    crow, ccol = (int)(rows/2), (int)(cols/2)
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.subplot(131),plt.imshow(breadboard_map, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
    
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getROIfromImage(src, offset = None):
    h, w, _ = src.shape

    image_center = None

    if offset:
        image_center = int(w/2 + 1) + offset[0], int(h/2 + 1) + offset[1]
    else:
        image_center = int(w/2 + 1), int(h/2 + 1)    

    x = int((w - image_center[0]) * SCALE)
    y = int((h - image_center[1]) * SCALE)

    return (image_center[0] - x, image_center[1] - y, image_center[0] + x, image_center[1] + y)

def test():
    breadboard_1_F = cv2.imread("images/Circuits/1_F.jpeg")
    # breadboard_map = cv2.imread("images/Breadboard/BreadboardMap.JPG")
    breadboard_map = cv2.imread("images/res.jpg") # grabcut으로 이쁘게 딴 브레드보드로 일단 테스트 할 거임
    pinmap = json.load(open("./static/data/pinmap.json", "r"))

    bY, bX, _ = pinmap['shape']
    tY, tX, _ = breadboard_map.shape

    breadboard_map = cv2.resize(breadboard_map, (bX, bY), interpolation=cv2.INTER_AREA)
    # breadboard_map = toBinary(breadboard_map)

    # 보통 한 0.7 ~ 0.8 정도 영역을 차지할 거임
    # [1]번이 높이, [0]번이 너비
    roi = getROIfromImage(breadboard_map,  offset=(0, -150))
    # breadboard_map = grab_cut(breadboard_map, roi)
    # breadboard_map = breadboard_map[roi[1]:roi[3], roi[0]:roi[2], :]
    # breadboard_map = toBinary(breadboard_map)

    y, x, _ = breadboard_map.shape
    new_line = int(0.66 * x/2)

    area = {
        '1': (0, int(breadboard_map.shape[1] / 2 - new_line - 1)),
        '2': (int(breadboard_map.shape[1] / 2 - new_line), int(breadboard_map.shape[1] / 2 - 1)),
        '3': (int(breadboard_map.shape[1] / 2), int(breadboard_map.shape[1] / 2 + new_line - 1)),
        '4': (int(breadboard_map.shape[1] / 2 + new_line), breadboard_map.shape[1])
    }
    pins = getBreadboardPin(breadboard_map, area)

    for pin in pins:
        start = pins[pin]['start']
        points = pins[pin]['points']

        for pts in points:
            x = int(start + pts[0])
            y = int(pts[1])
            cv2.circle(breadboard_map, (x, y), 15, (255, 255, 0), 10, cv2.FILLED)

    for i, pin in zip(['1', '2', '3', '4'], pinmap):
        start = pinmap[i]['start']
        points = pinmap[i]['points']

        for pts in points:
            x = int(start + pts[0])
            y = int(pts[1])
            cv2.circle(breadboard_map, (x, y), 10, (255, 0, 255), 10, cv2.FILLED)

    cv2.line(breadboard_map, (int(breadboard_map.shape[1] / 2), 0), (int(breadboard_map.shape[1] / 2), breadboard_map.shape[0]), (0, 0, 255), 10, cv2.LINE_AA)
    cv2.line(breadboard_map, (int(breadboard_map.shape[1] / 2 - new_line), 0), (int(breadboard_map.shape[1] / 2 - new_line), breadboard_map.shape[0]), (0, 0, 255), 10, cv2.LINE_AA)
    cv2.line(breadboard_map, (int(breadboard_map.shape[1] / 2 + new_line), 0), (int(breadboard_map.shape[1] / 2 + new_line), breadboard_map.shape[0]), (0, 0, 255), 10, cv2.LINE_AA)
    cv2.imwrite('./images/precess_image.jpg', breadboard_map)

if __name__ == "__main__":
    # getClearBreadboardMap()
    test()