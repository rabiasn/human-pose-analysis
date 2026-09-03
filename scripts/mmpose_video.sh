#!/bin/bash

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$#" -lt 1 ]; then
    echo "Kullanim:"
    echo "  ./scripts/mmpose_video.sh videos/input/walking_test.mp4"
    exit 1
fi

INPUT="$1"

if [[ "$INPUT" = /* ]]; then
    INPUT_ABS="$INPUT"
else
    INPUT_ABS="$PROJECT_ROOT/$INPUT"
fi

if [ ! -f "$INPUT_ABS" ]; then
    echo "Video bulunamadi: $INPUT_ABS"
    exit 1
fi

STEM="$(basename "$INPUT_ABS")"
STEM="${STEM%.*}"

PYTHON="$PROJECT_ROOT/.venv_mmpose/bin/python"

MMPOSE_ROOT="$PROJECT_ROOT/external/mmpose"

MMPose2D_SCRIPT="$PROJECT_ROOT/scripts/mmpose_video_2d.py"
JSON_TO_CSV_SCRIPT="$PROJECT_ROOT/scripts/mmpose_3d_json_to_csv.py"

OUT_3D_DIR="$PROJECT_ROOT/outputs/mmpose_video/${STEM}_3d"

DET_CONFIG="$MMPOSE_ROOT/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
DET_CHECKPOINT="$PROJECT_ROOT/models/mmpose/rtmdet_m_person.pth"

POSE2D_CONFIG="$MMPOSE_ROOT/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py"
POSE2D_CHECKPOINT="$PROJECT_ROOT/models/mmpose/rtmpose-m_coco_256x192.pth"

POSE3D_CONFIG="$MMPOSE_ROOT/configs/body_3d_keypoint/image_pose_lift/h36m/image-pose-lift_tcn_8xb64-200e_h36m.py"
POSE3D_CHECKPOINT="$PROJECT_ROOT/models/mmpose/simple3d_h36m.pth"

echo
echo "[1/3] MMPose 2D video islemi"
echo

"$PYTHON" "$MMPose2D_SCRIPT" "$INPUT_ABS"

echo
echo "[2/3] MMPose 3D pose lifting"
echo

mkdir -p "$OUT_3D_DIR"

cd "$MMPOSE_ROOT"

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
"$PYTHON" demo/body3d_pose_lifter_demo.py \
"$DET_CONFIG" \
"$DET_CHECKPOINT" \
"$POSE2D_CONFIG" \
"$POSE2D_CHECKPOINT" \
"$POSE3D_CONFIG" \
"$POSE3D_CHECKPOINT" \
--input "$INPUT_ABS" \
--output-root "$OUT_3D_DIR" \
--save-predictions \
--device cpu

echo
echo "[3/3] MMPose 3D JSON -> CSV"
echo

"$PYTHON" "$JSON_TO_CSV_SCRIPT" \
"$OUT_3D_DIR/results_${STEM}.json" \
"$PROJECT_ROOT/outputs/keypoints/mmpose/${STEM}_3d_keypoints.csv" \
--fps 30

echo
echo "MMPose video pipeline tamamlandi."
echo
echo "2D video:"
echo "  $PROJECT_ROOT/outputs/mmpose_video/${STEM}_2d_pose.mp4"
echo
echo "2D CSV:"
echo "  $PROJECT_ROOT/outputs/keypoints/mmpose/${STEM}_2d_keypoints.csv"
echo
echo "3D video:"
echo "  $OUT_3D_DIR/${STEM}.mp4"
echo
echo "3D JSON:"
echo "  $OUT_3D_DIR/results_${STEM}.json"
echo
echo "3D CSV:"
echo "  $PROJECT_ROOT/outputs/keypoints/mmpose/${STEM}_3d_keypoints.csv"
