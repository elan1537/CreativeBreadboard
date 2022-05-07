import random
import numpy as np
import cv2
import json

IMAGE_FILE = "./images/Circuits/220421/IMG_8534.JPG"
JSON_FILE = "./images/Circuits/220421/IMG_8534.json"

if __name__ == "__main__":
    img = cv2.imread(IMAGE_FILE, cv2.COLOR_BGR2RGB)

    segments = json.load(open(JSON_FILE))

    for label in segments['shapes']:
        points = label['points']

        R = int(random.random() * 255)
        G = int(random.random() * 255)
        B = int(random.random() * 255)

        if label['shape_type'] in ['line', 'linestrip']:
            img_blk = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
            for i in range(len(points)-1):
                x = tuple(np.array(points[i], dtype='int64'))
                y = tuple(np.array(points[i+1], dtype='int64'))
                cv2.line(img_blk, x, y, color=(B, G, R), thickness=60, lineType=cv2.LINE_4)

            contours, _ = cv2.findContours(img_blk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            d = np.array(contours, dtype=np.uint32)

            print(d[0][0], d[0][-1])
            print(len(points) * 2)

            print(d.shape)
            
            cv2.drawContours(img, contours, -1, color=(B, G, R), thickness=10, lineType=cv2.LINE_AA, maxLevel=1)
            break


    cv2.imshow("Segments", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()