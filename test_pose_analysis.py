import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from movement_rules import analyze_pose_2d, analyze_pose_3d

IMAGE = "images/test.jpg"
MODEL = "models/mediapipe/pose_landmarker_full.task"

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL),
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1
)

with vision.PoseLandmarker.create_from_options(options) as detector:
    image = mp.Image.create_from_file(IMAGE)
    result = detector.detect(image)

if not result.pose_landmarks:
    raise RuntimeError("2D poz bulunamadi.")

if not result.pose_world_landmarks:
    raise RuntimeError("3D poz bulunamadi.")

lm2d = result.pose_landmarks[0]
lm3d = result.pose_world_landmarks[0]

points_2d = {
    "nose":           (lm2d[0].x, lm2d[0].y),

    "left_shoulder":  (lm2d[11].x, lm2d[11].y),
    "right_shoulder": (lm2d[12].x, lm2d[12].y),

    "left_elbow":     (lm2d[13].x, lm2d[13].y),
    "right_elbow":    (lm2d[14].x, lm2d[14].y),

    "left_wrist":     (lm2d[15].x, lm2d[15].y),
    "right_wrist":    (lm2d[16].x, lm2d[16].y),

    "left_hip":       (lm2d[23].x, lm2d[23].y),
    "right_hip":      (lm2d[24].x, lm2d[24].y),

    "left_knee":      (lm2d[25].x, lm2d[25].y),
    "right_knee":     (lm2d[26].x, lm2d[26].y),

    "left_ankle":     (lm2d[27].x, lm2d[27].y),
    "right_ankle":    (lm2d[28].x, lm2d[28].y),
}

points_3d = {
    "left_shoulder":  (lm3d[11].x, lm3d[11].y, lm3d[11].z),
    "right_shoulder": (lm3d[12].x, lm3d[12].y, lm3d[12].z),
    "left_hip":       (lm3d[23].x, lm3d[23].y, lm3d[23].z),
    "right_hip":      (lm3d[24].x, lm3d[24].y, lm3d[24].z),
}

result_2d = analyze_pose_2d(points_2d)
result_3d = analyze_pose_3d(points_3d)

print("=== 2D DURUMLAR ===")
for key, value in result_2d.items():
    print(f"{key}: {value}")

print("\n=== 3D DURUM ===")
for key, value in result_3d.items():
    print(f"{key}: {value}")
