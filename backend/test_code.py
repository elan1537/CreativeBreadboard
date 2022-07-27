import server
import pytest
import json
import cv2
import base64, numpy as np
from findComponents import toPerspectiveImage

@pytest.fixture
def api():
    return server.app.test_client()

def test_detection(api):
    PADDING = 0
    scale = 0.25
    V = 15
    # target_image = cv2.imread("../IMG_5633.JPG", cv2.IMREAD_COLOR)
    # points = [[93,29],[99,871],[648,865],[648,27]]

    # target_image = cv2.imread("../IMG_5632.JPG", cv2.IMREAD_COLOR)
    # points = [[84,64],[100,882],[634,873],[626,53]]

    # target_image = cv2.imread("../IMG_5634.JPG", cv2.IMREAD_COLOR)
    # points = [[135,174],[124,860],[568,869],[595,188]]

    # target_image = cv2.imread("../IMG_5618.JPG", cv2.IMREAD_COLOR)
    # points = [[104,57],[95,945],[681,949],[686,59]]

    target_image = cv2.imread("../IMG_5607.JPG", cv2.IMREAD_COLOR)
    points = [[128,133],[117,921],[630,936],[657,139]]

    # target_image = cv2.imread("../IMG_5615.JPG", cv2.IMREAD_COLOR)
    # points = [[153,114],[162,861],[648,849],[641,111]]

    # target_image = cv2.imread("../IMG_5625.JPG", cv2.IMREAD_COLOR)
    # points = [[160,142],[158,834],[610,841],[621,144]]

    # target_image = cv2.imread("../images/Circuits/220428/Circuit-6.220428.jpg")
    # points = [[147,136],[147,796],[581,795],[580,133]]

    # target_image = cv2.imread("../images/Circuits/220428/Circuit-16.220428.jpg")
    # points = [[117,115],[111,891],[626,892],[626,118]]

    pts = []
    for point in points:
        pts.append([int(point[0] / scale), int(point[1] / scale)])

    base_point, res = toPerspectiveImage(target_image, np.array(pts), PADDING)

    _, buffer = cv2.imencode('.jpg', res)
    jpg_as_text = base64.b64encode(buffer).decode()

    payloads = {
        'pts': base_point.tolist(),
        'img_res': jpg_as_text,
        'scale': scale
    }

    resp = api.post('/detect', json=json.dumps(payloads), content_type='application/json')

    print(resp)