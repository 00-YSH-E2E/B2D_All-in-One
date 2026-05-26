#!/bin/bash
# GenAD B2D eval — Town04 + TensorRT backend (backbone TRT 8.6, head PT fallback). 자동 dump.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
export E2E_BACKEND=trt
export B2D_MODEL=${B2D_MODEL:-/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/Model}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/tmp/trt86_sdk/lib
export GENAD_DUMP_DIR=${GENAD_DUMP_DIR-/tmp/genad_dump}
export GENAD_DUMP_TAG=${GENAD_DUMP_TAG:-town04_trt}
export GENAD_DUMP_MAX=${GENAD_DUMP_MAX:-50}
echo "[genad_trt_town04] B2D_MODEL=$B2D_MODEL"
echo "[genad_trt_town04] LD_LIBRARY_PATH first entry: ${LD_LIBRARY_PATH%%:*}"
echo "[genad_trt_town04] DUMP=$GENAD_DUMP_DIR/$GENAD_DUMP_TAG (max $GENAD_DUMP_MAX steps)"
echo "[genad_trt_town04] NOTE: head TRT engine not built (TRT 8.6 ONNX parser limit) — head runs PT fallback."
exec bash "$HERE/run_evaluation_multi_genad_town04.sh" "$@"
