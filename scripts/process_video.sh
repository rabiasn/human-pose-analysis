#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Kullanim: $0 <video_yolu> [openpose_net_resolution]"
    exit 1
fi

INPUT="$1"
NET_RESOLUTION="${2:--1x256}"

if [[ "$INPUT" != /* ]]; then
    INPUT="$ROOT/$INPUT"
fi

if [[ ! -f "$INPUT" ]]; then
    echo "HATA: Video bulunamadi: $INPUT"
    exit 1
fi

MEDIAPIPE_PY="$ROOT/.venv_mediapipe/bin/python"
MEDIAPIPE_SCRIPT="$ROOT/scripts/mediapipe_video.py"
OPENPOSE_SCRIPT="$ROOT/scripts/openpose_video.sh"

echo "========================================"
echo "VIDEO: $INPUT"
echo "========================================"

echo
echo "[1/3] MediaPipe basliyor..."
"$MEDIAPIPE_PY" "$MEDIAPIPE_SCRIPT" "$INPUT"

echo
echo "[2/3] OpenPose basliyor..."
"$OPENPOSE_SCRIPT" "$INPUT" "$NET_RESOLUTION"

STEM="$(basename "$INPUT")"
STEM="${STEM%.*}"

echo
echo "[3/3] OpenPose temporal duzeltme basliyor..."
"$MEDIAPIPE_PY" "$ROOT/scripts/openpose_temporal.py"     "$ROOT/outputs/keypoints/openpose/$STEM"     "$ROOT/outputs/keypoints/openpose/${STEM}_temporal.csv"

echo
echo "========================================"
echo "TUM ISLEMLER TAMAMLANDI"
echo "========================================"
