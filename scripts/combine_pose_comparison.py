import cv2
import numpy as np

videos = [
    ("MediaPipe", "outputs/mediapipe_video/baby_test_10s_pose.mp4"),
    ("OpenPose", "outputs/openpose_video/baby_test_10s_pose.mp4"),
    ("MMPose", "outputs/mmpose_video/baby_test_10s_2d_pose.mp4"),
    ("DeeperCut", "outputs/deepercut_multicut/baby_test_10s_deepercut_pose.mp4"),
]

caps = [cv2.VideoCapture(path) for _, path in videos]

W, H = 360, 640
FPS = 30
OUT = "outputs/baby_test_10s_comparison.mp4"

writer = cv2.VideoWriter(
    OUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (W * 2, H * 2)
)

def fit(frame):
    h, w = frame.shape[:2]
    scale = min(W / w, H / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (nw, nh))

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    x = (W - nw) // 2
    y = (H - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas

for _ in range(300):
    panels = []

    for (name, _), cap in zip(videos, caps):
        ok, frame = cap.read()
        if not ok:
            break

        panel = fit(frame)

        cv2.rectangle(panel, (0, 0), (W, 55), (0, 0, 0), -1)
        cv2.putText(
            panel,
            name,
            (15, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        panels.append(panel)

    if len(panels) != 4:
        break

    top = np.hstack((panels[0], panels[1]))
    bottom = np.hstack((panels[2], panels[3]))
    combined = np.vstack((top, bottom))

    writer.write(combined)

for cap in caps:
    cap.release()

writer.release()

print("Hazır:", OUT)
