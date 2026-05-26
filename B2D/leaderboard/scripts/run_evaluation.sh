#!/bin/bash
# CARLA_ROOT 는 env override 가능하도록 변경 (예전엔 YOUR_CARLA_PATH 가 박혀 있어 즉시 실패).
export CARLA_ROOT=${CARLA_ROOT:-/home/Humble/Carla/Carla_0.9.15}
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
# CARLA Python egg — b2d_zoo (py3.8) 환경 기준 기본값. PY_TAG env 로 override 가능.
export PY_TAG=${PY_TAG:-py3.8}
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-${PY_TAG}-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
# GenAD agent (vad_b2d_agent.py) lives under Bench2DriveZoo/team_code/ via the
# B2D/Bench2DriveZoo symlink. Need both the team_code dir (so the agent module
# resolves) and the B2D root (so 'from Bench2DriveZoo.team_code.* import ...'
# inside the agent works).
export PYTHONPATH=$PYTHONPATH:Bench2DriveZoo/team_code
export PYTHONPATH=$PYTHONPATH:.
export PYTHONPATH=$PYTHONPATH:scenario_runner
export SCENARIO_RUNNER_ROOT=scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=$1
export TM_PORT=$2
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export RESUME=True
export IS_BENCH2DRIVE=$3
export PLANNER_TYPE=$9
export GPU_RANK=${10}

# TCP evaluation
export ROUTES=$4
export TEAM_AGENT=$5
export TEAM_CONFIG=$6
export CHECKPOINT_ENDPOINT=$7
export SAVE_PATH=$8

CUDA_VISIBLE_DEVICES=${GPU_RANK} python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--traffic-manager-port=${TM_PORT} \
--gpu-rank=${GPU_RANK} \
