#!/bin/bash
# GenAD B2D eval — TensorRT backend (backbone TRT 8.6 fp16, head는 TRT 8.6 parser 한계로 PT fallback)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
export E2E_BACKEND=trt
export B2D_MODEL=${B2D_MODEL:-/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/Model}

# TRT 8.6 wheel uses libnvinfer.so.8 from /tmp/trt86_sdk (extracted from local .deb).
# Without this LD_LIBRARY_PATH the libplugins.so / tensorrt python wheel may pick up
# system libnvinfer.so.10 and fail.
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/tmp/trt86_sdk/lib

echo "[genad_trt] B2D_MODEL=$B2D_MODEL"
echo "[genad_trt] LD_LIBRARY_PATH first entry: ${LD_LIBRARY_PATH%%:*}"
echo "[genad_trt] NOTE: head TRT engine not built (ONNX parser If/Slice limit in TRT 8.6) — head runs PT fallback."
echo "[genad_trt]       To enable full TRT head, use TRT 10 trtexec on the head ONNX."
exec bash "$HERE/run_evaluation_multi_genad.sh" "$@"
