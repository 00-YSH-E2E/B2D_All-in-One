#!/bin/bash
# DT B2D eval — Town04 시나리오만 (bench2drive220 중 town="Town04" 인 12개 route).
# 호출자가 export 한 E2E_BACKEND/B2D_MODEL 사용.
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220_town04
TEAM_AGENT=team_code/drivetransformer_b2d_agent.py

export B2D_MODEL=${B2D_MODEL:-/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/Model}
export E2E_BACKEND=${E2E_BACKEND:-pytorch}

# DT 의 'team_code' / 'mmcv' / 'DriveTransformer' 패키지가 B2D/DriveTransformer 심링크를 통해
# 해석되도록 PYTHONPATH 를 prepend. (Bench2DriveZoo == GenAD mmcv 와의 mmcv 충돌 회피)
export PYTHONPATH=DriveTransformer:DriveTransformer/team_code:$PYTHONPATH

TEAM_CONFIG=DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py+${B2D_MODEL}/drivetransformer/pth/drivetransformer_large.pth
BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220_town04
PLANNER_TYPE=only_traj
ALGO=DriveTransformer_${E2E_BACKEND}_town04
SAVE_PATH=./eval_bench2drive220_${ALGO}_${PLANNER_TYPE}

if [ ! -d "${ALGO}_b2d_${PLANNER_TYPE}" ]; then
    mkdir ${ALGO}_b2d_${PLANNER_TYPE}
fi

# split 은 pre-split 했어도 flag 만들어 둠 (재실행 시 split skip)
SPLIT_FLAG="${BASE_ROUTES}_DriveTransformer_${PLANNER_TYPE}_split_done.flag"
if [ ! -f "$SPLIT_FLAG" ]; then
    echo -e "\033[33m Running split_xml.py on Town04 subset for DT \033[0m"
    TASK_NUM=${TASK_NUM:-2}
    python tools/split_xml.py $BASE_ROUTES $TASK_NUM DriveTransformer $PLANNER_TYPE
    touch "$SPLIT_FLAG"
else
    echo -e "\033[32m Splitting already done. \033[0m"
fi

GPU_RANK_LIST=${GPU_RANK_LIST:-"0"}
TASK_LIST=${TASK_LIST:-"0"}
echo -e "\033[32m E2E_BACKEND: $E2E_BACKEND \033[0m"
echo -e "\033[32m B2D_MODEL:   $B2D_MODEL \033[0m"
echo -e "\033[32m PYTHONPATH (head): ${PYTHONPATH%%:*}, ... \033[0m"
echo -e "\033[32m GPU_RANK_LIST: $GPU_RANK_LIST \033[0m"
echo -e "\033[32m TASK_LIST:     $TASK_LIST \033[0m"
echo -e "***********************************************************************************"

read -ra GPU_RANKS <<< "$GPU_RANK_LIST"
read -ra TASKS     <<< "$TASK_LIST"
length=${#GPU_RANKS[@]}
for ((i=0; i<$length; i++ )); do
    PORT=$((BASE_PORT + i * 150))
    TM_PORT=$((BASE_TM_PORT + i * 150))
    ROUTES="${BASE_ROUTES}_${TASKS[$i]}_DriveTransformer_${PLANNER_TYPE}.xml"
    CHECKPOINT_ENDPOINT="${ALGO}_b2d_${PLANNER_TYPE}/${BASE_CHECKPOINT_ENDPOINT}_${TASKS[$i]}.json"
    GPU_RANK=${GPU_RANKS[$i]}
    echo -e "\033[32m TASK_ID: ${TASKS[$i]}  PORT: $PORT  TM_PORT: $TM_PORT  GPU: $GPU_RANK \033[0m"
    echo -e "\033[32m ROUTES:  $ROUTES \033[0m"
    echo -e "\033[32m CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT \033[0m"
    echo -e "***********************************************************************************"
    bash -e leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK 2>&1 > ${BASE_ROUTES}_${TASKS[$i]}_DriveTransformer_${E2E_BACKEND}_${PLANNER_TYPE}.log &
    sleep 5
done
wait
