import numpy as np
import cv2
import sys

image_path = sys.argv[1]
npz_path = sys.argv[2]
out_path = sys.argv[3]

img = cv2.imread(image_path)
pose = np.load(npz_path)["pose"]

points = [(int(pose[0, i]), int(pose[1, i])) for i in range(14)]
conf = pose[2]

edges = [
    (0,1), (1,2), (2,3), (3,4), (4,5),
    (6,7), (7,8), (8,12), (12,9), (9,10), (10,11),
    (12,13)
]

for a, b in edges:
    if conf[a] > 0.1 and conf[b] > 0.1:
        cv2.line(img, points[a], points[b], (255,255,255), 3)

for i, (x, y) in enumerate(points):
    if conf[i] > 0.1:
        cv2.circle(img, (x, y), 5, (0,0,255), -1)

cv2.imwrite(out_path, img)
print(out_path)
