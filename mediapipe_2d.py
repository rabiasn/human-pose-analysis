import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections

IMAGE_PATH = "images/test.jpg"
MODEL_PATH = "models/mediapipe/pose_landmarker_full.task"
OUTPUT_PATH = "outputs/mediapipe_test_rendered.jpg"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)

with vision.PoseLandmarker.create_from_options(options) as detector:

    image = mp.Image.create_from_file(IMAGE_PATH)
    result = detector.detect(image)

    frame = cv2.imread(IMAGE_PATH)
    h, w = frame.shape[:2]

    print("Bulunan kisi:", len(result.pose_landmarks))

    for pose_landmarks in result.pose_landmarks:

        # İskelet çizgileri
        for connection in PoseLandmarksConnections.POSE_LANDMARKS:
            p1 = pose_landmarks[connection.start]
            p2 = pose_landmarks[connection.end]

            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 33 landmark noktası
        for i, landmark in enumerate(pose_landmarks):
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            print(
                f"{i:2d} "
                f"x={landmark.x:.4f} "
                f"y={landmark.y:.4f} "
                f"z={landmark.z:.4f}"
            )

cv2.imwrite(OUTPUT_PATH, frame)

print("Cikti:", OUTPUT_PATH)
