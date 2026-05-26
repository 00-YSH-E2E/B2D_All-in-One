#!/bin/bash
# GenAD B2D eval — PyTorch backend (baseline)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
export E2E_BACKEND=pytorch
export B2D_MODEL=${B2D_MODEL:-/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/Model}
echo "[genad_pytorch] B2D_MODEL=$B2D_MODEL"
exec bash "$HERE/run_evaluation_multi_genad.sh" "$@"
