#!/bin/bash
# GenAD B2D eval — ONNX backend (backbone ORT, head는 TRT-only plugin op 때문에 PT fallback)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
export E2E_BACKEND=onnx
export B2D_MODEL=${B2D_MODEL:-/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/Model}
echo "[genad_onnx] B2D_MODEL=$B2D_MODEL"
echo "[genad_onnx] NOTE: head ONNX uses TRT custom ops (RotatePlugin) — head runs PT fallback."
exec bash "$HERE/run_evaluation_multi_genad.sh" "$@"
