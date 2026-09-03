#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="MMPose pose-lifter JSON sonucunu frame bazli CSV'ye cevirir."
    )

    parser.add_argument(
        "input_json",
        help="MMPose body3d_pose_lifter_demo.py tarafindan uretilen JSON"
    )

    parser.add_argument(
        "output_csv",
        nargs="?",
        default=None,
        help="Cikti CSV yolu"
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video FPS degeri. Default: 30"
    )

    return parser.parse_args()


def resolve_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()


def get_keypoint_name_map(meta_info):
    raw = meta_info.get("keypoint_id2name", {})

    result = {}

    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                result[int(key)] = str(value)
            except (TypeError, ValueError):
                pass

    return result


def main():
    args = parse_args()

    input_json = resolve_path(args.input_json)

    if not input_json.is_file():
        raise FileNotFoundError(
            f"JSON bulunamadi: {input_json}"
        )

    if args.fps <= 0:
        raise ValueError("FPS 0'dan buyuk olmali.")

    if args.output_csv is None:
        output_csv = (
            PROJECT_ROOT
            / "outputs"
            / "keypoints"
            / "mmpose"
            / f"{input_json.stem}_3d_keypoints.csv"
        )
    else:
        output_csv = resolve_path(args.output_csv)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        input_json,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    if "instance_info" not in data:
        raise KeyError(
            "JSON icinde 'instance_info' bulunamadi."
        )

    meta_info = data.get("meta_info", {})
    frames = data["instance_info"]

    keypoint_names = get_keypoint_name_map(
        meta_info
    )

    expected_keypoints = int(
        meta_info.get("num_keypoints", 17)
    )

    total_frames = 0
    detected_frames = 0
    missing_frames = 0
    bad_shape_frames = []

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "frame",
            "source_frame_id",
            "timestamp_ms",
            "keypoint_id",
            "keypoint_name",
            "x",
            "y",
            "z",
            "lifter_score",
        ])

        for frame_entry in frames:
            source_frame_id = int(
                frame_entry.get(
                    "frame_id",
                    total_frames + 1
                )
            )

            # MMPose demo 1-based frame_id kullaniyor.
            # Projenin diger CSV'leri ile uyum icin 0-based yapiliyor.
            frame_idx = source_frame_id - 1

            timestamp_ms = (
                frame_idx / args.fps
            ) * 1000.0

            instances = frame_entry.get(
                "instances",
                []
            )

            total_frames += 1

            if not instances:
                missing_frames += 1

                for keypoint_id in range(
                    expected_keypoints
                ):
                    writer.writerow([
                        frame_idx,
                        source_frame_id,
                        round(timestamp_ms, 3),
                        keypoint_id,
                        keypoint_names.get(
                            keypoint_id,
                            f"keypoint_{keypoint_id}"
                        ),
                        "",
                        "",
                        "",
                        "",
                    ])

                continue

            detected_frames += 1

            # Bu proje videosunda tek ana kisi kullaniliyor.
            instance = instances[0]

            keypoints = instance.get(
                "keypoints",
                []
            )

            lifter_scores = instance.get(
                "keypoint_scores",
                []
            )

            valid_shape = (
                len(keypoints)
                == expected_keypoints
                and all(
                    isinstance(point, list)
                    and len(point) == 3
                    for point in keypoints
                )
            )

            if not valid_shape:
                bad_shape_frames.append(
                    frame_idx
                )
                continue

            for keypoint_id, point in enumerate(
                keypoints
            ):
                x, y, z = point

                if keypoint_id < len(
                    lifter_scores
                ):
                    lifter_score = (
                        lifter_scores[
                            keypoint_id
                        ]
                    )
                else:
                    lifter_score = ""

                writer.writerow([
                    frame_idx,
                    source_frame_id,
                    round(timestamp_ms, 3),
                    keypoint_id,
                    keypoint_names.get(
                        keypoint_id,
                        f"keypoint_{keypoint_id}"
                    ),
                    round(float(x), 6),
                    round(float(y), 6),
                    round(float(z), 6),
                    (
                        round(
                            float(lifter_score),
                            6
                        )
                        if lifter_score != ""
                        else ""
                    ),
                ])

    print(
        "MMPose 3D JSON -> CSV tamamlandi."
    )
    print(
        f"Toplam frame       : {total_frames}"
    )
    print(
        f"3D pose bulunan    : {detected_frames}"
    )
    print(
        f"3D pose bulunamayan: {missing_frames}"
    )
    print(
        f"Bozuk shape frame  : {bad_shape_frames}"
    )
    print(
        f"CSV                : {output_csv}"
    )

    print()
    print(
        "Not: lifter_score RTMPose 2D confidence "
        "degildir. Pose-lifter ciktisindaki skordur."
    )


if __name__ == "__main__":
    main()
