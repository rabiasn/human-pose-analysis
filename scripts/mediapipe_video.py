import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Giris video yolu")
    parser.add_argument(
        "--model",
        default="models/mediapipe/pose_landmarker_full.task"
    )
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    input_path = Path(args.video)
    stem = input_path.stem

    output_video = Path(
        args.output_video
        or f"outputs/mediapipe_video/{stem}_pose.mp4"
    )
    output_csv = Path(
        args.output_csv
        or f"outputs/keypoints/mediapipe/{stem}_keypoints.csv"
    )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Video yazici acilamadi: {output_video}")

    base_options = python.BaseOptions(
        model_asset_path=args.model
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )

    with open(output_csv, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame",
            "timestamp_ms",
            "landmark_id",
            "x",
            "y",
            "z",
            "visibility",
            "presence"
        ])

        with vision.PoseLandmarker.create_from_options(options) as detector:
            frame_index = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp_ms = int(round(frame_index * 1000.0 / fps))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                result = detector.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                if not result.pose_landmarks:
                    for landmark_id in range(33):
                        csv_writer.writerow([
                            frame_index,
                            timestamp_ms,
                            landmark_id,
                            "",
                            "",
                            "",
                            "",
                            ""
                        ])

                for pose_landmarks in result.pose_landmarks:
                    for connection in PoseLandmarksConnections.POSE_LANDMARKS:
                        p1 = pose_landmarks[connection.start]
                        p2 = pose_landmarks[connection.end]

                        x1 = int(p1.x * width)
                        y1 = int(p1.y * height)
                        x2 = int(p2.x * width)
                        y2 = int(p2.y * height)

                        cv2.line(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                    for landmark_id, landmark in enumerate(pose_landmarks):
                        x_px = int(landmark.x * width)
                        y_px = int(landmark.y * height)

                        cv2.circle(
                            frame,
                            (x_px, y_px),
                            4,
                            (0, 0, 255),
                            -1
                        )

                        csv_writer.writerow([
                            frame_index,
                            timestamp_ms,
                            landmark_id,
                            landmark.x,
                            landmark.y,
                            landmark.z,
                            landmark.visibility,
                            landmark.presence
                        ])

                writer.write(frame)

                if frame_index % 30 == 0:
                    print(f"Frame: {frame_index}")

                frame_index += 1

    cap.release()
    writer.release()

    print("Tamamlandi")
    print("Video:", output_video)
    print("Keypoints:", output_csv)


if __name__ == "__main__":
    main()
