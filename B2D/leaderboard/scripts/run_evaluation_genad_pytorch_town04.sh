#!/bin/bash
# GenAD B2D eval — Town04 + PyTorch backend. GENAD_DUMP_DIR 켜면 자동으로 dump.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
export E2E_BACKEND=pytorch
export B2D_MODEL=${B2D_MODEL:-/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/Model}
# 디폴트로 dump 활성화 — 끄려면 GENAD_DUMP_DIR= (빈값) 으로 호출.
export GENAD_DUMP_DIR=${GENAD_DUMP_DIR-/tmp/genad_dump}
export GENAD_DUMP_TAG=${GENAD_DUMP_TAG:-town04_pytorch}
export GENAD_DUMP_MAX=${GENAD_DUMP_MAX:-50}
echo "[genad_pytorch_town04] B2D_MODEL=$B2D_MODEL"
echo "[genad_pytorch_town04] DUMP=$GENAD_DUMP_DIR/$GENAD_DUMP_TAG (max $GENAD_DUMP_MAX steps)"
exec bash "$HERE/run_evaluation_multi_genad_town04.sh" "$@"
