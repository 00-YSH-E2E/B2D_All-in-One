#!/bin/bash
# DT B2D eval — Town04 + PyTorch backend (DT large, fp32). DT 는 backend adapter 미적용 → 항상 PT.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
export E2E_BACKEND=pytorch
export B2D_MODEL=${B2D_MODEL:-/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/Model}
echo "[dt_pytorch_town04] B2D_MODEL=$B2D_MODEL"
echo "[dt_pytorch_town04] ckpt=$B2D_MODEL/drivetransformer/pth/drivetransformer_large.pth"
exec bash "$HERE/run_evaluation_multi_dt_town04.sh" "$@"
