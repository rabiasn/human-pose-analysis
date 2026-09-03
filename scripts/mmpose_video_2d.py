#!/usr/bin/env python3

import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import adapt_mmdet_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MMPOSE_ROOT = PROJECT_ROOT / "external" / "mmpose"

DET_CONFIG = (
    MMPOSE_ROOT
    / "demo"
    / "mmdetection_cfg"
    / "rtmdet_m_640-8xb32_coco-person.py"
)

DET_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "mmpose"
    / "rtmdet_m_person.pth"
)

POSE_CONFIG = (
    MMPOSE_ROOT
    / "configs"
    / "body_2d_keypoint"
    / "rtmpose"
    / "coco"
    / "rtmpose-m_8xb256-420e_coco-256x192.py"
)

POSE_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "mmpose"
    / "rtmpose-m_coco_256x192.pth"
)


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),

    (5, 6),

    (5, 7),
    (7, 9),

    (6, 8),
    (8, 10),

    (5, 11),
    (6, 12),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="MMPose RTMDet + RTMPose video 2D pose extraction"
    )

    parser.add_argument(
        "input",
        help="Input video path"
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device. Default: cpu"
    )

    parser.add_argument(
        "--bbox-thr",
        type=float,
        default=0.3,
        help="Person detection confidence threshold"
    )

    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=0.3,
        help="Keypoint drawing threshold"
    )

    return parser.parse_args()


def validate_files(paths):
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Dosya bulunamadi: {path}")


def draw_pose(frame, keypoints, scores, bbox, bbox_score, kpt_thr):
    output = frame.copy()

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"person {bbox_score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    for a, b in SKELETON:
        if scores[a] < kpt_thr or scores[b] < kpt_thr:
            continue

        p1 = tuple(np.round(keypoints[a]).astype(int))
        p2 = tuple(np.round(keypoints[b]).astype(int))

        cv2.line(
            output,
            p1,
            p2,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    for keypoint_id, (point, score) in enumerate(zip(keypoints, scores)):
        if score < kpt_thr:
            continue

        x, y = np.round(point).astype(int)

        cv2.circle(
            output,
            (x, y),
            4,
            (0, 0, 255),
            -1,
            cv2.LINE_AA
        )

    return output


def write_missing_frame(csv_writer, frame_idx, timestamp_ms):
    for keypoint_id, keypoint_name in enumerate(KEYPOINT_NAMES):
        csv_writer.writerow([
            frame_idx,
            round(timestamp_ms, 3),
            keypoint_id,
            keypoint_name,
            "",
            "",
            "",
            "",
        ])


def main():
    args = parse_args()

    input_path = Path(args.input)

    if not input_path.is_absolute():
        input_path = (PROJECT_ROOT / input_path).resolve()

    validate_files([
        input_path,
        DET_CONFIG,
        DET_CHECKPOINT,
        POSE_CONFIG,
        POSE_CHECKPOINT,
    ])

    stem = input_path.stem

    output_video_dir = PROJECT_ROOT / "outputs" / "mmpose_video"
    output_csv_dir = PROJECT_ROOT / "outputs" / "keypoints" / "mmpose"

    output_video_dir.mkdir(parents=True, exist_ok=True)
    output_csv_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_video_dir / f"{stem}_2d_pose.mp4"
    output_csv = output_csv_dir / f"{stem}_2d_keypoints.csv"

    print("RTMDet yukleniyor...")

    detector = init_detector(
        str(DET_CONFIG),
        str(DET_CHECKPOINT),
        device=args.device
    )

    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    print("RTMPose-M yukleniyor...")

    pose_estimator = init_model(
        str(POSE_CONFIG),
        str(POSE_CHECKPOINT),
        device=args.device
    )

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise RuntimeError("Video FPS bilgisi okunamadi.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Output video acilamadi: {output_video}")

    csv_file = open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    )

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "frame",
        "timestamp_ms",
        "keypoint_id",
        "keypoint_name",
        "x",
        "y",
        "keypoint_score",
        "bbox_score",
    ])

    frame_idx = 0
    detected_frames = 0
    missing_frames = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            timestamp_ms = (frame_idx / fps) * 1000.0

            det_result = inference_detector(
                detector,
                frame
            )

            pred = det_result.pred_instances.cpu().numpy()

            valid_mask = (
                (pred.labels == 0)
                & (pred.scores > args.bbox_thr)
            )

            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) == 0:
                missing_frames += 1

                write_missing_frame(
                    csv_writer,
                    frame_idx,
                    timestamp_ms
                )

                writer.write(frame)

                frame_idx += 1
                continue

            best_index = valid_indices[
                np.argmax(pred.scores[valid_indices])
            ]

            bbox = pred.bboxes[best_index]
            bbox_score = float(pred.scores[best_index])

            pose_results = inference_topdown(
                pose_estimator,
                frame,
                bbox[None, :]
            )

            if not pose_results:
                missing_frames += 1

                write_missing_frame(
                    csv_writer,
                    frame_idx,
                    timestamp_ms
                )

                writer.write(frame)

                frame_idx += 1
                continue

            pose_instance = (
                pose_results[0]
                .pred_instances
                .cpu()
                .numpy()
            )

            keypoints = pose_instance.keypoints[0]
            keypoint_scores = pose_instance.keypoint_scores[0]

            if keypoints.shape != (17, 2):
                raise RuntimeError(
                    f"Beklenmeyen keypoint shape: {keypoints.shape}"
                )

            if keypoint_scores.shape != (17,):
                raise RuntimeError(
                    "Beklenmeyen keypoint score shape: "
                    f"{keypoint_scores.shape}"
                )

            detected_frames += 1

            for keypoint_id in range(17):
                x = float(keypoints[keypoint_id][0])
                y = float(keypoints[keypoint_id][1])
                score = float(keypoint_scores[keypoint_id])

                csv_writer.writerow([
                    frame_idx,
                    round(timestamp_ms, 3),
                    keypoint_id,
                    KEYPOINT_NAMES[keypoint_id],
                    round(x, 6),
                    round(y, 6),
                    round(score, 6),
                    round(bbox_score, 6),
                ])

            vis_frame = draw_pose(
                frame=frame,
                keypoints=keypoints,
                scores=keypoint_scores,
                bbox=bbox,
                bbox_score=bbox_score,
                kpt_thr=args.kpt_thr,
            )

            writer.write(vis_frame)

            frame_idx += 1

            if frame_idx % 25 == 0 or frame_idx == total_frames:
                print(
                    f"Frame {frame_idx}/{total_frames}"
                )

    finally:
        cap.release()
        writer.release()
        csv_file.close()

    print()
    print("MMPose 2D video islemi tamamlandi.")
    print(f"Toplam frame       : {frame_idx}")
    print(f"Kisi bulunan frame : {detected_frames}")
    print(f"Kisi bulunamayan   : {missing_frames}")
    print(f"Video              : {output_video}")
    print(f"CSV                : {output_csv}")


if __name__ == "__main__":
    main()
