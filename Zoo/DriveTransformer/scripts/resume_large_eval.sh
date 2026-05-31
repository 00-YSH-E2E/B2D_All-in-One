#!/bin/bash
# 대형맵 165 중 미완분(PT 91→165, TRT 105→165)을 resume로 이어서 완성. CARLA 크래시 대비 재시도 루프.
# 기존 json(large_eval_*.json) 안 지움 — leaderboard --resume 이 완료 route 건너뛰고 이어감.
B=/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive
RES=$B/Result/Drivetransformer
STATUS=$RES/OVERNIGHT_STATUS.txt
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $1" | tee -a "$STATUS"; }
kill_carla(){ for pid in $(pgrep -f CarlaUE4 2>/dev/null); do c=$(cat /proc/$pid/comm 2>/dev/null); case "$c" in CarlaUE4*) kill -9 "$pid" 2>/dev/null;; esac; done; sleep 5; }
count(){ python3 -c "import json;print(len(json.load(open('$1'))['_checkpoint']['records']))" 2>/dev/null || echo 0; }

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
ROUTES=leaderboard/data/large_eval.xml

[ -f "$ROUTES" ] || python3 - <<'PY'
import xml.etree.ElementTree as ET
root=ET.parse("/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/B2D/leaderboard/data/bench2drive220.xml").getroot()
BIG={'Town11','Town12','Town13','Town15'}
sel=[r for r in root.findall('route') if r.get('town') in BIG]
new=ET.Element('routes'); [new.append(r) for r in sel]
ET.ElementTree(new).write("/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/B2D/leaderboard/data/large_eval.xml",encoding='utf-8',xml_declaration=True)
PY

log "RESUME START — 대형맵 이어서 (PT $(count large_eval_pt.json)/165, TRT $(count large_eval_trt.json)/165)"

# resume_phase <backend> <json> <tag> <dbk> <log>
resume_phase(){
  local BK=$1 OUT=$2 TAG=$3 DBK=$4 LOG=$5 MEM=""
  [ "$BK" = trt ] && MEM="DT_TRT_MEMORY=1"
  local stall=0
  while true; do
    local n=$(count "$OUT")
    [ "$n" -ge 165 ] && { log "  $TAG 완료 ($n/165)"; break; }
    log "  $TAG 진행 $n/165 — run_evaluation(resume) 시도 (stall=$stall)"
    kill_carla
    env E2E_BACKEND="$BK" $MEM DT_DEBUG_BACKEND="$DBK" DT_DEBUG_TAG="$TAG" \
      bash leaderboard/scripts/run_evaluation.sh 30000 50000 True "$ROUTES" "$AGENT" "$TEAM_CONFIG" \
      "$OUT" "./ckpt_$TAG" only_traj 0 >> "leaderboard/data/$LOG" 2>&1
    local after=$(count "$OUT")
    if [ "$after" -le "$n" ]; then stall=$((stall+1)); else stall=0; fi
    [ "$stall" -ge 3 ] && { log "  $TAG 진척없음 3회 연속 → 중단 ($after/165, 특정 route 반복 크래시 의심)"; break; }
  done
}

resume_phase pytorch large_eval_pt.json  large_pt  Pytorch  ovB_pt_resume.log
resume_phase trt     large_eval_trt.json large_trt TensorRT ovB_trt_resume.log

# 채점표 재생성 (대형 + 220 병합)
kill_carla
python3 DriveTransformer/scripts/scorecard.py large_eval_pt.json large_eval_trt.json > "$RES/SCORECARD_LARGE.txt" 2>&1
log "대형맵 채점표 갱신 → SCORECARD_LARGE.txt"
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
log "전체 220 채점표 갱신 → SCORECARD_FULL.txt"
log "RESUME ALL DONE — PT $(count large_eval_pt.json)/165, TRT $(count large_eval_trt.json)/165"