import numpy as np
import cv2
import json
import os
import base64
import io

CONV_TO_DO = "./images/Circuits/220421"
WORKING_DIR = os.path.join(CONV_TO_DO, "polygons")

if __name__ == "__main__":
    if not os.path.isdir(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    files_to_do = [json for json in os.listdir(CONV_TO_DO) if ".json" in json]


    for js in files_to_do:
        json_data = json.load(open(os.path.join(CONV_TO_DO, js), "r"))
        origin_shape = (json_data["imageHeight"], json_data["imageWidth"])
        

        if js == "IMG_8559-modify.json":
            image = cv2.imdecode(
                np.frombuffer(
                    base64.b64decode(json_data["imageData"]), np.uint8), 
                    cv2.IMREAD_COLOR
                )
            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        structure_label = [e for e in json_data["shapes"] if "structure" in e["label"]]

        json_result = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": json_data["imagePath"],
            "imageData": json_data["imageData"],
            "imageHeight": json_data["imageHeight"],
            "imageWidth": json_data["imageWidth"]
        }

        for e in structure_label:
            space = np.zeros(shape=(origin_shape[0], origin_shape[1], 1), dtype="uint8")
            pts = e["points"]

            for idx in range(len(pts)):
                if idx+1 == len(pts):
                    break

                pt_s = (int(pts[idx][0]), int(pts[idx][1]))
                pt_e = (int(pts[idx+1][0]), int(pts[idx+1][1]))
                cv2.line(space, pt_s, pt_e, (255, 255, 255), 80)

            _, space = cv2.threshold(space, -1, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(space, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            t = {}
            t["label"] = e["label"]
            t["points"] = []
            t["group_id"] = e["group_id"]
            t["shape_type"] = "polygon"
            t["flags"] = e["flags"]

            for contour in contours:
                p = cv2.arcLength(contour, True)
                approximatedShape = cv2.approxPolyDP(contour, 0.001 * p, True)

                for i in range(len(approximatedShape)):
                    if i+1 == len(approximatedShape):
                        break

                    pt_s = approximatedShape[i][0].tolist()
                    t["points"].append(pt_s)

            json_result["shapes"].append(t)

        js = js.replace(".json", "")
        file_name = f"{js}-modify.json"
        with open(os.path.join(WORKING_DIR, file_name), "w") as f:
            json.dump(json_result, f)