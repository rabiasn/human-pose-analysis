import argparse
import os
from pathlib import Path
import csv
from collections import defaultdict

import cv2

BODY_25_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (10, 11),
    (8, 12), (12, 13), (13, 14),
    (0, 15), (15, 17),
    (0, 16), (16, 18),
    (14, 19), (19, 20), (14, 21),
    (11, 22), (22, 23), (11, 24),
]

STATUS_COLORS = {
    "visible": (0, 255, 0),
    "interpolated": (0, 255, 0),
}


def load_points(csv_path):
    frames = defaultdict(dict)

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            kid = int(row["keypoint_id"])
            status = row["status"]

            if row["x"] == "" or row["y"] == "":
                frames[frame][kid] = {"x": None, "y": None, "status": status}
            else:
                frames[frame][kid] = {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "status": status,
                }

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("temporal_csv")
    parser.add_argument("output_video")
    args = parser.parse_args()

    points_by_frame = load_points(args.temporal_csv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = Path(args.output_video)
    temp_output = output_path.with_name(output_path.stem + "_tmp.mp4")

    writer = cv2.VideoWriter(
        str(temp_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Video yazici acilamadi: {args.output_video}")

    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pts = points_by_frame.get(frame_idx, {})

        for a, b in BODY_25_PAIRS:
            pa = pts.get(a)
            pb = pts.get(b)

            if not pa or not pb:
                continue
            if pa["x"] is None or pb["x"] is None:
                continue

            if pa["status"] == "interpolated" or pb["status"] == "interpolated":
                color = STATUS_COLORS["interpolated"]
            else:
                color = STATUS_COLORS["visible"]

            cv2.line(
                frame,
                (int(pa["x"]), int(pa["y"])),
                (int(pb["x"]), int(pb["y"])),
                color,
                2
            )

        for _, p in pts.items():
            if p["x"] is None:
                continue

            color = STATUS_COLORS.get(p["status"], (255, 255, 255))
            cv2.circle(
                frame,
                (int(p["x"]), int(p["y"])),
                4,
                color,
                -1
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    os.replace(temp_output, output_path)

    print("Cikti:", output_path)


if __name__ == "__main__":
    main()
