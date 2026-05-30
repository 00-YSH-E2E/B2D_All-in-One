#!/bin/bash
# DT PT vs TRT 전체 평가(표준맵 55 route, 40종 시나리오) + 채점표 — overnight 자동 실행.
# 진행: Result/Drivetransformer/OVERNIGHT_STATUS.txt, 결과: SCORECARD.txt + run_NNN(NPZ/viz).
B=/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive
STATUS=$B/Result/Drivetransformer/OVERNIGHT_STATUS.txt
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $1" | tee -a "$STATUS"; }
kill_carla(){ for pid in $(pgrep -f CarlaUE4 2>/dev/null); do c=$(cat /proc/$pid/comm 2>/dev/null); [ "$c" = "CarlaUE4-Linux" ] && kill -9 "$pid" 2>/dev/null; done; sleep 4; }

source /home/Humble/Setting/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo
cd "$B/B2D" || exit 1
export PYTHONPATH=DriveTransformer:DriveTransformer/team_code:$PYTHONPATH
export B2D_MODEL=$B/Model
export GENAD_DUMP_DIR=
export DT_DEBUG_DIR=$B/Result/Drivetransformer
export DT_DEBUG_SAVE_EVERY=50
TEAM_CONFIG="DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py+$B2D_MODEL/drivetransformer/pth/drivetransformer_large.pth"
AGENT=team_code/drivetransformer_b2d_agent.py
ROUTES=leaderboard/data/full_eval.xml

: > "$STATUS"
log "START — PT vs TRT 전체 평가 (표준맵)"

# --- route 파일 (Town01-07,10HD = 55, Town13 대형맵 제외) ---
python3 - <<'PY'
import xml.etree.ElementTree as ET
root=ET.parse("/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/B2D/leaderboard/data/bench2drive220.xml").getroot()
OK={'Town01','Town02','Town03','Town04','Town05','Town06','Town07','Town10HD'}
sel=[r for r in root.findall('route') if r.get('town') in OK]
new=ET.Element('routes'); [new.append(r) for r in sel]
ET.ElementTree(new).write("/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/B2D/leaderboard/data/full_eval.xml",
                          encoding='utf-8', xml_declaration=True)
PY
log "route 파일 빌드: $(grep -c '<route ' $ROUTES) routes"

# --- PT 전체 ---
log "PT eval 시작"
kill_carla
rm -f full_eval_pt.json
E2E_BACKEND=pytorch DT_DEBUG_BACKEND=Pytorch DT_DEBUG_TAG=full_pt \
  bash leaderboard/scripts/run_evaluation.sh 30000 50000 True "$ROUTES" "$AGENT" "$TEAM_CONFIG" \
  full_eval_pt.json ./fe_pt only_traj 0 > leaderboard/data/overnight_pt.log 2>&1
log "PT eval 완료 (records=$(python3 -c "import json;print(len(json.load(open('full_eval_pt.json'))['_checkpoint']['records']))" 2>/dev/null))"

# --- TRT 전체 (memory ON) ---
log "TRT eval 시작"
kill_carla
rm -f full_eval_trt.json
E2E_BACKEND=trt DT_TRT_MEMORY=1 DT_DEBUG_BACKEND=TensorRT DT_DEBUG_TAG=full_trt \
  bash leaderboard/scripts/run_evaluation.sh 30000 50000 True "$ROUTES" "$AGENT" "$TEAM_CONFIG" \
  full_eval_trt.json ./fe_trt only_traj 0 > leaderboard/data/overnight_trt.log 2>&1
log "TRT eval 완료 (records=$(python3 -c "import json;print(len(json.load(open('full_eval_trt.json'))['_checkpoint']['records']))" 2>/dev/null))"

# --- 채점표 ---
kill_carla
python3 DriveTransformer/scripts/scorecard.py full_eval_pt.json full_eval_trt.json \
  > "$B/Result/Drivetransformer/SCORECARD.txt" 2>&1
log "채점표 생성 완료 → Result/Drivetransformer/SCORECARD.txt"
log "ALL DONE"
