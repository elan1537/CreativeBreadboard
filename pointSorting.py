import json
import numpy as np

if __name__ == "__main__":
    pinmap = json.load(open("./static/data/pinmap.json", "r"))


    for pin in ["2", "3"]:
        target = pinmap[pin]
        points = np.uint32(sorted(target["points"])).reshape(5, -1, 2)
        
        temp = []
        for group in points:
            group = sorted(group.tolist(), key = lambda x: [x[1]])
            temp.append(group)

        pinmap[pin]["points"] = np.uint32(temp).reshape(-1, 2).tolist()

    for pin in ["1", "4"]:
        target = pinmap[pin]
        points = np.uint32(sorted(target["points"])).reshape(2, -1, 2)

        temp = []
        for group in points:
            group = sorted(group.tolist(), key = lambda x: [x[1]])
            temp.append(group)

        pinmap[pin]["points"] = np.uint32(temp).reshape(-1, 2).tolist()

    with open("pinmap.json", "w") as f:
        json.dump(pinmap, f)

