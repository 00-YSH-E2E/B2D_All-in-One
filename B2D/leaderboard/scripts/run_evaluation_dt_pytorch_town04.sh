#!/bin/bash
# DT B2D eval — Town04 + PyTorch backend (DT large, fp32). DT 는 backend adapter 미적용 → 항상 PT.
# GENAD_DUMP_DIR (이름은 공유) 켜면 매 step input_data_batch + 모델 출력을 NPZ 로 저장.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
export E2E_BACKEND=pytorch
export B2D_MODEL=${B2D_MODEL:-/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/Model}
# 기본 dump 활성화 — 끄려면 GENAD_DUMP_DIR= (빈값) 으로 호출.
export GENAD_DUMP_DIR=${GENAD_DUMP_DIR-/tmp/genad_dump}
export GENAD_DUMP_TAG=${GENAD_DUMP_TAG:-town04_dt_pytorch}
export GENAD_DUMP_MAX=${GENAD_DUMP_MAX:-50}
echo "[dt_pytorch_town04] B2D_MODEL=$B2D_MODEL"
echo "[dt_pytorch_town04] ckpt=$B2D_MODEL/drivetransformer/pth/drivetransformer_large.pth"
echo "[dt_pytorch_town04] DUMP=$GENAD_DUMP_DIR/$GENAD_DUMP_TAG (max $GENAD_DUMP_MAX steps)"
exec bash "$HERE/run_evaluation_multi_dt_town04.sh" "$@"
