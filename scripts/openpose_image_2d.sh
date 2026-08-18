#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p outputs/openpose_run

./external/openpose/build/examples/openpose/openpose.bin \
  --image_dir images \
  --model_folder external/openpose/models \
  --model_pose BODY_25 \
  --display 0 \
  --write_images outputs/openpose_run \
  --write_json outputs/openpose_run
