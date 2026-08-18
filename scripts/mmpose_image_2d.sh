#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT/external/mmpose"

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python demo/image_demo.py \
  ../../images/test.jpg \
  configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py \
  ../../models/mmpose/rtmpose-m_coco_256x192.pth \
  --out-file ../../outputs/mmpose_2d.jpg \
  --device cpu
