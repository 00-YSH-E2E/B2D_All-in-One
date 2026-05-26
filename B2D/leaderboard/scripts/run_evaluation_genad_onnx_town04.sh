#!/bin/bash
# GenAD B2D eval — Town04 + ONNX backend (backbone ORT, head PT fallback). 자동 dump.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
export E2E_BACKEND=onnx
export B2D_MODEL=${B2D_MODEL:-/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/Model}
export GENAD_DUMP_DIR=${GENAD_DUMP_DIR-/tmp/genad_dump}
export GENAD_DUMP_TAG=${GENAD_DUMP_TAG:-town04_onnx}
export GENAD_DUMP_MAX=${GENAD_DUMP_MAX:-50}
echo "[genad_onnx_town04] B2D_MODEL=$B2D_MODEL"
echo "[genad_onnx_town04] DUMP=$GENAD_DUMP_DIR/$GENAD_DUMP_TAG (max $GENAD_DUMP_MAX steps)"
echo "[genad_onnx_town04] NOTE: head ONNX uses TRT custom ops — head runs PT fallback."
exec bash "$HERE/run_evaluation_multi_genad_town04.sh" "$@"
