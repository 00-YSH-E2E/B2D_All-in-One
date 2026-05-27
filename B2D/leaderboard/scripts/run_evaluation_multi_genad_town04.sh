#!/bin/bash
# GenAD B2D eval — Town04 시나리오만 (bench2drive220 중 town="Town04" 인 12개 route).
# E2E_BACKEND 와 GENAD_DUMP_DIR 는 호출자가 export 한 값을 그대로 사용.
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220_town04
TEAM_AGENT=team_code/vad_b2d_agent.py

export B2D_MODEL=${B2D_MODEL:-/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/Model}
export E2E_BACKEND=${E2E_BACKEND:-pytorch}

TEAM_CONFIG=Bench2DriveZoo/adzoo/genad/configs/VAD/GenAD_config_b2d.py+${B2D_MODEL}/genad/pth/epoch_6.pth
BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220_town04
PLANNER_TYPE=traj
ALGO=genad_${E2E_BACKEND}_town04
SAVE_PATH=./eval_bench2drive220_${ALGO}_${PLANNER_TYPE}

if [ ! -d "${ALGO}_b2d_${PLANNER_TYPE}" ]; then
    mkdir ${ALGO}_b2d_${PLANNER_TYPE}
fi

# Town04 12 routes 를 task 단위로 쪼갬 — Town04 전용 split flag.
SPLIT_FLAG="${BASE_ROUTES}_genad_${PLANNER_TYPE}_split_done.flag"
if [ ! -f "$SPLIT_FLAG" ]; then
    echo -e "\033[33m Running split_xml.py on Town04 subset \033[0m"
    TASK_NUM=${TASK_NUM:-2}   # 12 routes 면 2 task 정도가 적당 (1 task = 6 routes)
    python tools/split_xml.py $BASE_ROUTES $TASK_NUM genad $PLANNER_TYPE
    touch "$SPLIT_FLAG"
else
    echo -e "\033[32m Splitting already done. \033[0m"
fi

# 기본: 단일 GPU 단일 task (12 routes 순차) — 디버깅에 적합. 빠르게 돌리려면
# GPU_RANK_LIST="0 1" TASK_LIST="0 1" 처럼 env override.
GPU_RANK_LIST=${GPU_RANK_LIST:-"0"}
TASK_LIST=${TASK_LIST:-"0"}
echo -e "\033[32m E2E_BACKEND:    $E2E_BACKEND \033[0m"
echo -e "\033[32m B2D_MODEL:      $B2D_MODEL \033[0m"
echo -e "\033[32m GENAD_DUMP_DIR: ${GENAD_DUMP_DIR:-<unset>} \033[0m"
echo -e "\033[32m GENAD_DUMP_TAG: ${GENAD_DUMP_TAG:-<unset>} \033[0m"
echo -e "\033[32m GENAD_DUMP_MAX: ${GENAD_DUMP_MAX:-<unset (default 50)>} \033[0m"
echo -e "\033[32m GPU_RANK_LIST:  $GPU_RANK_LIST \033[0m"
echo -e "\033[32m TASK_LIST:      $TASK_LIST \033[0m"
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
    echo -e "\033[32m TASK_ID: ${TASKS[$i]}  PORT: $PORT  TM_PORT: $TM_PORT  GPU: $GPU_RANK \033[0m"
    echo -e "\033[32m ROUTES:  $ROUTES \033[0m"
    echo -e "\033[32m CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT \033[0m"
    echo -e "***********************************************************************************"
    bash -e leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK > ${BASE_ROUTES}_${TASKS[$i]}_genad_${E2E_BACKEND}_${PLANNER_TYPE}.log 2>&1 &
    sleep 5
done
wait
