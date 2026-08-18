import mediapipe as mp
import matplotlib.pyplot as plt

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections

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

if not result.pose_world_landmarks:
    raise RuntimeError("3D landmark bulunamadi.")

points = result.pose_world_landmarks[0]

xs = [p.x for p in points]
ys = [p.y for p in points]
zs = [p.z for p in points]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

# Landmark noktaları
ax.scatter(xs, zs, [-y for y in ys], s=35)

# İskelet bağlantıları
for connection in PoseLandmarksConnections.POSE_LANDMARKS:
    a = points[connection.start]
    b = points[connection.end]

    ax.plot(
        [a.x, b.x],
        [a.z, b.z],
        [-a.y, -b.y],
        linewidth=2
    )

ax.set_xlabel("X (metre) - sag/sol")
ax.set_ylabel("Z (metre) - derinlik")
ax.set_zlabel("Y (metre) - yukari/asagi")

ax.set_title("MediaPipe Pose - 3D World Landmarks")

# Eksenleri yaklaşık eşit ölçekle
max_range = max(
    max(xs) - min(xs),
    max(zs) - min(zs),
    max([-y for y in ys]) - min([-y for y in ys])
) / 2

mid_x = (max(xs) + min(xs)) / 2
mid_z = (max(zs) + min(zs)) / 2
vertical = [-y for y in ys]
mid_y = (max(vertical) + min(vertical)) / 2

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_z - max_range, mid_z + max_range)
ax.set_zlim(mid_y - max_range, mid_y + max_range)

plt.tight_layout()
plt.savefig("outputs/mediapipe_3d.png", dpi=200)

print("3D landmark sayisi:", len(points))
print("Kaydedildi: outputs/mediapipe_3d.png")
print("Pencerede fareyle surukleyerek 3D modeli dondurebilirsin.")

plt.show()
