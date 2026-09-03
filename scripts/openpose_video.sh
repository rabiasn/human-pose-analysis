#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OPENPOSE_BIN="$ROOT/external/openpose/build/examples/openpose/openpose.bin"
MODEL_FOLDER="$ROOT/external/openpose/models"

if [[ $# -lt 1 ]]; then
    echo "Kullanim: $0 <video_yolu> [net_resolution]"
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

if [[ ! -x "$OPENPOSE_BIN" ]]; then
    echo "HATA: OpenPose binary bulunamadi: $OPENPOSE_BIN"
    exit 1
fi

STEM="$(basename "$INPUT")"
STEM="${STEM%.*}"

OUTPUT_VIDEO="$ROOT/outputs/openpose_video/${STEM}_pose.mp4"
OUTPUT_JSON="$ROOT/outputs/keypoints/openpose/$STEM"

mkdir -p "$ROOT/outputs/openpose_video"
rm -rf "$OUTPUT_JSON"
mkdir -p "$OUTPUT_JSON"
rm -f "$OUTPUT_VIDEO"

echo "Video      : $INPUT"
echo "Net çöz.   : $NET_RESOLUTION"
echo "Video çıktı: $OUTPUT_VIDEO"
echo "JSON çıktı : $OUTPUT_JSON"

"$OPENPOSE_BIN" \
    --video "$INPUT" \
    --model_folder "$MODEL_FOLDER" \
    --net_resolution "$NET_RESOLUTION" \
    --write_video "$OUTPUT_VIDEO" \
    --write_json "$OUTPUT_JSON" \
    --display 0 \
    --render_pose 1

echo "Tamamlandi."
