#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT/external/mmpose"

TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python demo/body3d_pose_lifter_demo.py \
  demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
  ../../models/mmpose/rtmdet_m_person.pth \
  configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py \
  ../../models/mmpose/rtmpose-m_coco_256x192.pth \
  configs/body_3d_keypoint/image_pose_lift/h36m/image-pose-lift_tcn_8xb64-200e_h36m.py \
  ../../models/mmpose/simple3d_h36m.pth \
  --input ../../images/test.jpg \
  --output-root ../../outputs/mmpose_3d_run \
  --save-predictions \
  --device cpu
