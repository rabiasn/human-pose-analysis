import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_dir")
    parser.add_argument("output_csv")
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.1)
    args = parser.parse_args()

    files = sorted(Path(args.json_dir).glob("*_keypoints.json"))

    frames = []

    for frame_idx, path in enumerate(files):
        data = json.loads(path.read_text())

        points = []

        if data.get("people"):
            kp = data["people"][0]["pose_keypoints_2d"]
        else:
            kp = [0.0] * 75

        for i in range(25):
            x, y, conf = kp[i * 3:i * 3 + 3]

            valid = (
                conf >= args.min_confidence
                and x != 0
                and y != 0
            )

            points.append({
                "x": x if valid else None,
                "y": y if valid else None,
                "confidence": conf,
                "status": "visible" if valid else "missing"
            })

        frames.append(points)

    # Kisa kayiplari lineer interpolation ile doldur
    for keypoint_id in range(25):
        i = 0

        while i < len(frames):
            if frames[i][keypoint_id]["status"] == "visible":
                i += 1
                continue

            start = i

            while (
                i < len(frames)
                and frames[i][keypoint_id]["status"] == "missing"
            ):
                i += 1

            end = i
            gap = end - start

            if (
                gap <= args.max_gap
                and start > 0
                and end < len(frames)
                and frames[start - 1][keypoint_id]["status"] == "visible"
                and frames[end][keypoint_id]["status"] == "visible"
            ):
                p0 = frames[start - 1][keypoint_id]
                p1 = frames[end][keypoint_id]

                for j in range(gap):
                    ratio = (j + 1) / (gap + 1)

                    p = frames[start + j][keypoint_id]

                    p["x"] = p0["x"] + ratio * (p1["x"] - p0["x"])
                    p["y"] = p0["y"] + ratio * (p1["y"] - p0["y"])
                    p["status"] = "interpolated"

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "frame",
            "keypoint_id",
            "x",
            "y",
            "confidence",
            "status"
        ])

        for frame_idx, points in enumerate(frames):
            for keypoint_id, p in enumerate(points):
                writer.writerow([
                    frame_idx,
                    keypoint_id,
                    "" if p["x"] is None else p["x"],
                    "" if p["y"] is None else p["y"],
                    p["confidence"],
                    p["status"]
                ])

    print("Cikti:", output)


if __name__ == "__main__":
    main()
