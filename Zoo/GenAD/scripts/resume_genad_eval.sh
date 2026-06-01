#!/bin/bash
# GenAD eval 재부팅/중단 복구 — 4 phase(full_pt/full_trt/large_pt/large_trt) 중 미완분만 resume.
# 기존 genad_*.json 안 지움 (leaderboard --resume이 완료 route 건너뜀). crash 재시도 루프 포함.
# TRT 주의는 run_full_eval_genad_overnight.sh 헤더 참고. TRT 건너뛰려면 SKIP_TRT=1.
B=/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive
RES=$B/Result/GenAD
STATUS=$RES/OVERNIGHT_STATUS.txt
mkdir -p "$RES"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $1" | tee -a "$STATUS"; }
kill_carla(){ for pid in $(pgrep -f CarlaUE4 2>/dev/null); do c=$(cat /proc/$pid/comm 2>/dev/null); case "$c" in CarlaUE4*) kill -9 "$pid" 2>/dev/null;; esac; done; sleep 5; }
count(){ python3 -c "import json;print(len(json.load(open('$1'))['_checkpoint']['records']))" 2>/dev/null || echo 0; }

source /home/Humble/Setting/miniconda3/etc/profile.d/conda.sh
conda activate b2d_zoo
cd "$B/B2D" || exit 1
export B2D_MODEL=$B/Model
export GENAD_DUMP_DIR=$RES/dump
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/tmp/trt86_sdk/lib
TEAM_CONFIG="Bench2DriveZoo/adzoo/genad/configs/VAD/GenAD_config_b2d.py+$B2D_MODEL/genad/pth/epoch_6.pth"
AGENT=team_code/vad_b2d_agent.py
PLANNER=traj
SKIP_TRT=${SKIP_TRT:-0}

log "RESUME START — GenAD 미완분 이어서 (full_pt $(count genad_full_pt.json) / full_trt $(count genad_full_trt.json) / large_pt $(count genad_large_pt.json) / large_trt $(count genad_large_trt.json))"

# resume_phase: rm 안 함, 재시도 루프. <backend> <routes> <outjson> <savepath> <logname>
resume_phase(){
  local BK=$1 ROUTES=$2 OUT=$3 SAVE=$4 LOG=$5
  local TGT=$(grep -c '<route ' "$ROUTES")
  [ "$BK" = trt ] && [ "$SKIP_TRT" = 1 ] && { log "  $(basename $OUT) — SKIP_TRT=1 건너뜀"; return; }
  local stall=0
  while true; do
    local n=$(count "$OUT")
    [ "$n" -ge "$TGT" ] && { log "  $(basename $OUT) 완료 ($n/$TGT)"; break; }
    log "  $(basename $OUT) 진행 $n/$TGT — run_evaluation(resume) 시도 (stall=$stall)"
    kill_carla
    env E2E_BACKEND="$BK" \
      bash leaderboard/scripts/run_evaluation.sh 30000 50000 True "$ROUTES" "$AGENT" "$TEAM_CONFIG" \
      "$OUT" "$SAVE" "$PLANNER" 0 >> "leaderboard/data/$LOG" 2>&1
    local after=$(count "$OUT")
    [ "$after" -le "$n" ] && stall=$((stall+1)) || stall=0
    [ "$stall" -ge 3 ] && { log "  $(basename $OUT) 진척없음 3회 → 중단 ($after/$TGT)"; break; }
  done
}

resume_phase pytorch leaderboard/data/full_eval.xml  genad_full_pt.json   ./gd_full_pt   ovGA_pt.log
resume_phase trt     leaderboard/data/full_eval.xml  genad_full_trt.json  ./gd_full_trt  ovGA_trt.log
resume_phase pytorch leaderboard/data/large_eval.xml genad_large_pt.json  ./gd_large_pt  ovGB_pt.log
resume_phase trt     leaderboard/data/large_eval.xml genad_large_trt.json ./gd_large_trt ovGB_trt.log

# 채점표 재생성
kill_carla
sc(){ python3 DriveTransformer/scripts/scorecard.py "$1" "$2" > "$3" 2>&1; }
sc genad_full_pt.json  genad_full_trt.json  "$RES/SCORECARD.txt"
sc genad_large_pt.json genad_large_trt.json "$RES/SCORECARD_LARGE.txt"
python3 - <<'PY'
import json
def merge(a,b,out):
    da=json.load(open(a)); db=json.load(open(b))
    da["_checkpoint"]["records"] += db["_checkpoint"]["records"]; json.dump(da, open(out,'w'))
B="/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/B2D/"
try:
    merge(B+"genad_full_pt.json",  B+"genad_large_pt.json",  B+"genad_all_pt.json")
    merge(B+"genad_full_trt.json", B+"genad_large_trt.json", B+"genad_all_trt.json")
    print("merged ok")
except Exception as e: print("merge skip:", e)
PY
[ -f genad_all_pt.json ] && [ -f genad_all_trt.json ] && sc genad_all_pt.json genad_all_trt.json "$RES/SCORECARD_FULL.txt"
log "RESUME ALL DONE — 채점표 갱신 완료"