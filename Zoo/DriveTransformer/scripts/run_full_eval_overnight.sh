#!/bin/bash
# DT 220 route 전체 PT vs TRT eval — 단일 순차 프로세스(race 없음). 표준55 → 대형165, 단계별+전체 채점표.
# 진행: Result/Drivetransformer/OVERNIGHT_STATUS.txt / 결과: SCORECARD.txt(55)·SCORECARD_LARGE.txt(165)·SCORECARD_FULL.txt(220)
B=/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive
RES=$B/Result/Drivetransformer
STATUS=$RES/OVERNIGHT_STATUS.txt
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $1" | tee -a "$STATUS"; }
# comm 15자 제한 때문에 'CarlaUE4*' prefix 매칭 (정확매칭 = 버그)
kill_carla(){ for pid in $(pgrep -f CarlaUE4 2>/dev/null); do c=$(cat /proc/$pid/comm 2>/dev/null); case "$c" in CarlaUE4*) kill -9 "$pid" 2>/dev/null;; esac; done; sleep 5; }

source /home/Humble/Setting/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo
cd "$B/B2D" || exit 1
export PYTHONPATH=DriveTransformer:DriveTransformer/team_code:$PYTHONPATH
export B2D_MODEL=$B/Model
export GENAD_DUMP_DIR=
export DT_DEBUG_DIR=$RES
export DT_DEBUG_SAVE_EVERY=50
TEAM_CONFIG="DriveTransformer/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py+$B2D_MODEL/drivetransformer/pth/drivetransformer_large.pth"
AGENT=team_code/drivetransformer_b2d_agent.py

: > "$STATUS"
log "START — DT 220 route 전체 PT vs TRT (단일 순차)"

# --- route 파일 2개: 표준55(full_eval) + 대형165(large_eval) ---
python3 - <<'PY'
import xml.etree.ElementTree as ET
root=ET.parse("/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/B2D/leaderboard/data/bench2drive220.xml").getroot()
STD={'Town01','Town02','Town03','Town04','Town05','Town06','Town07','Town10HD'}
BIG={'Town11','Town12','Town13','Town15'}
def write(sel,path):
    new=ET.Element('routes'); [new.append(r) for r in sel]
    ET.ElementTree(new).write(path, encoding='utf-8', xml_declaration=True)
write([r for r in root.findall('route') if r.get('town') in STD],
      "/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/B2D/leaderboard/data/full_eval.xml")
write([r for r in root.findall('route') if r.get('town') in BIG],
      "/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/B2D/leaderboard/data/large_eval.xml")
PY
log "route 빌드: 표준 $(grep -c '<route ' leaderboard/data/full_eval.xml) / 대형 $(grep -c '<route ' leaderboard/data/large_eval.xml)"

# run_one <backend> <routes> <outjson> <debugtag> <debugbackend> <logname>
run_one(){
  local BK=$1 ROUTES=$2 OUT=$3 TAG=$4 DBK=$5 LOG=$6
  kill_carla; rm -f "$OUT"
  local MEM=""; [ "$BK" = "trt" ] && MEM="DT_TRT_MEMORY=1"
  env E2E_BACKEND="$BK" $MEM DT_DEBUG_BACKEND="$DBK" DT_DEBUG_TAG="$TAG" \
    bash leaderboard/scripts/run_evaluation.sh 30000 50000 True "$ROUTES" "$AGENT" "$TEAM_CONFIG" \
    "$OUT" "./ckpt_$TAG" only_traj 0 > "leaderboard/data/$LOG" 2>&1
  log "  $TAG 완료 (records=$(python3 -c "import json;print(len(json.load(open('$OUT'))['_checkpoint']['records']))" 2>/dev/null))"
}

# ===== Phase A: 표준 55 =====
log "[Phase A] 표준 55 — PT 시작"
run_one pytorch leaderboard/data/full_eval.xml full_eval_pt.json  full_pt  Pytorch   ovA_pt.log
log "[Phase A] 표준 55 — TRT 시작"
run_one trt     leaderboard/data/full_eval.xml full_eval_trt.json full_trt TensorRT  ovA_trt.log
kill_carla
python3 DriveTransformer/scripts/scorecard.py full_eval_pt.json full_eval_trt.json > "$RES/SCORECARD.txt" 2>&1
log "[Phase A] 채점표 → SCORECARD.txt"

# ===== Phase B: 대형 165 =====
log "[Phase B] 대형 165 — PT 시작"
run_one pytorch leaderboard/data/large_eval.xml large_eval_pt.json  large_pt  Pytorch  ovB_pt.log
log "[Phase B] 대형 165 — TRT 시작"
run_one trt     leaderboard/data/large_eval.xml large_eval_trt.json large_trt TensorRT ovB_trt.log
kill_carla
python3 DriveTransformer/scripts/scorecard.py large_eval_pt.json large_eval_trt.json > "$RES/SCORECARD_LARGE.txt" 2>&1
log "[Phase B] 채점표 → SCORECARD_LARGE.txt"

# ===== 전체 220 병합 채점표 =====
python3 - <<'PY'
import json
def merge(a,b,out):
    da=json.load(open(a)); db=json.load(open(b))
    da["_checkpoint"]["records"] += db["_checkpoint"]["records"]
    json.dump(da, open(out,'w'))
B="/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/B2D/"
try:
    merge(B+"full_eval_pt.json",  B+"large_eval_pt.json",  B+"all_pt.json")
    merge(B+"full_eval_trt.json", B+"large_eval_trt.json", B+"all_trt.json")
    print("merged ok")
except Exception as e:
    print("merge 실패:", e)
PY
python3 DriveTransformer/scripts/scorecard.py all_pt.json all_trt.json > "$RES/SCORECARD_FULL.txt" 2>&1
log "전체 220 채점표 → SCORECARD_FULL.txt"
log "ALL DONE — 220 route 전체 eval 완료"