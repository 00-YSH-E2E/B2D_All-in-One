#!/bin/bash
# GenAD B2D 멀티 GPU eval 런처 — backend는 E2E_BACKEND 환경변수로 결정 (pytorch/onnx/trt)
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220
TEAM_AGENT=team_code/vad_b2d_agent.py

# B2D_MODEL은 Bench2Drive/Model 트리. agent의 backend adapter가 이 경로에서 엔진/ONNX 찾음.
export B2D_MODEL=${B2D_MODEL:-/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/Model}
export E2E_BACKEND=${E2E_BACKEND:-pytorch}

# GenAD config + 체크포인트 (B2D_MODEL/genad/pth/epoch_6.pth 가 정본)
TEAM_CONFIG=Bench2DriveZoo/adzoo/genad/configs/VAD/GenAD_config_b2d.py+${B2D_MODEL}/genad/pth/epoch_6.pth
BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220
PLANNER_TYPE=traj
ALGO=genad_${E2E_BACKEND}
SAVE_PATH=./eval_bench2drive220_${ALGO}_${PLANNER_TYPE}

if [ ! -d "${ALGO}_b2d_${PLANNER_TYPE}" ]; then
    mkdir ${ALGO}_b2d_${PLANNER_TYPE}
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} created. \033[0m"
else
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} already exists. \033[0m"
fi

# Use a 'genad' (not 'genad_<backend>') split flag — splits are backend-independent.
SPLIT_FLAG="${BASE_ROUTES}_genad_${PLANNER_TYPE}_split_done.flag"
if [ ! -f "$SPLIT_FLAG" ]; then
    echo -e "****************************\033[33m Attention \033[0m ****************************"
    echo -e "\033[33m Running split_xml.py \033[0m"
    TASK_NUM=${TASK_NUM:-8}
    python tools/split_xml.py $BASE_ROUTES $TASK_NUM genad $PLANNER_TYPE
    touch "$SPLIT_FLAG"
    echo -e "\033[32m Splitting complete. Flag file created. \033[0m"
else
    echo -e "\033[32m Splitting already done. \033[0m"
fi

echo -e "**************\033[36m Please Manually adjust GPU or TASK_ID \033[0m **************"
# Default: single GPU, single task — change to (0 1 ... 7) for full eval.
GPU_RANK_LIST=${GPU_RANK_LIST:-"0"}
TASK_LIST=${TASK_LIST:-"0"}
echo -e "\033[32m E2E_BACKEND: $E2E_BACKEND \033[0m"
echo -e "\033[32m B2D_MODEL:   $B2D_MODEL \033[0m"
echo -e "\033[32m GPU_RANK_LIST: $GPU_RANK_LIST \033[0m"
echo -e "\033[32m TASK_LIST:     $TASK_LIST \033[0m"
echo -e "***********************************************************************************"

read -ra GPU_RANKS <<< "$GPU_RANK_LIST"
read -ra TASKS     <<< "$TASK_LIST"
length=${#GPU_RANKS[@]}
for ((i=0; i<$length; i++ )); do
    PORT=$((BASE_PORT + i * 150))
    TM_PORT=$((BASE_TM_PORT + i * 150))
    ROUTES="${BASE_ROUTES}_${TASKS[$i]}_genad_${PLANNER_TYPE}.xml"
    CHECKPOINT_ENDPOINT="${ALGO}_b2d_${PLANNER_TYPE}/${BASE_CHECKPOINT_ENDPOINT}_${TASKS[$i]}.json"
    GPU_RANK=${GPU_RANKS[$i]}
    echo -e "\033[32m ALGO: $ALGO \033[0m"
    echo -e "\033[32m PLANNER_TYPE: $PLANNER_TYPE \033[0m"
    echo -e "\033[32m TASK_ID: ${TASKS[$i]} \033[0m"
    echo -e "\033[32m PORT: $PORT \033[0m"
    echo -e "\033[32m TM_PORT: $TM_PORT \033[0m"
    echo -e "\033[32m CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT \033[0m"
    echo -e "\033[32m GPU_RANK: $GPU_RANK \033[0m"
    echo -e "\033[32m bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK \033[0m"
    echo -e "***********************************************************************************"
    bash -e leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK > ${BASE_ROUTES}_${TASKS[$i]}_genad_${E2E_BACKEND}_${PLANNER_TYPE}.log 2>&1 &
    sleep 5
done
wait
