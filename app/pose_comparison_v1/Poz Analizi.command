#!/bin/zsh
cd "$(dirname "$0")/../.." || exit 1
printf '\nPoz Analizi açılıyor. Tarayıcı adresi: http://127.0.0.1:8765\n\n'
exec .venv_mediapipe/bin/python -B app/pose_comparison_v1/server.py --port 8765
