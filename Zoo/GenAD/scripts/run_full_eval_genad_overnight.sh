#!/bin/bash
# GenAD 220 route 전체 PT vs TRT eval (DT의 run_full_eval_overnight 미러 + crash 재시도 루프 내장).
# 표준55(Town01-07,10HD) → 대형165(Town11/12/13/15), 각 PT→TRT, 단계별+전체 채점표.
#
# ⚠️ TRT 주의 — GenAD TRT는 TRT 8.6 기반(backbone만 TRT fp16, head는 ONNX parser 한계로 PT fallback).
#    /tmp/trt86_sdk/lib(libnvinfer.so.8) 필요. 그런데 현재 env엔 DT용으로 올린 TRT 10.8 python wheel이
#    깔려 있어 GenAD TRT backbone 엔진(8.6) 로드가 깨질 수 있음. TRT phase를 돌리려면 먼저:
#        pip install tensorrt-cu11==8.6.1   (※ 이러면 DT TRT가 깨지지만 DT eval은 이미 끝나 무방)
#    PyTorch backend는 env와 무관하게 바로 됨 → 우선 PT부터 검증 권장.
#    TRT 건너뛰려면: SKIP_TRT=1 환경변수로 실행.
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
export GENAD_DUMP_DIR=$RES/dump          # 디버깅 dump (무거우면 빈값으로: export GENAD_DUMP_DIR=)
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/tmp/trt86_sdk/lib   # GenAD TRT 8.6 libs
TEAM_CONFIG="Bench2DriveZoo/adzoo/genad/configs/VAD/GenAD_config_b2d.py+$B2D_MODEL/genad/pth/epoch_6.pth"
AGENT=team_code/vad_b2d_agent.py
PLANNER=traj
SKIP_TRT=${SKIP_TRT:-0}

: > "$STATUS"
log "START — GenAD 220 route 전체 PT vs TRT (SKIP_TRT=$SKIP_TRT)"

# route 파일 (DT가 만든 표준55/대형165 재사용, 없으면 재생성)
build_routes(){ python3 - <<'PY'
import xml.etree.ElementTree as ET
root=ET.parse("/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/B2D/leaderboard/data/bench2drive220.xml").getroot()
STD={'Town01','Town02','Town03','Town04','Town05','Town06','Town07','Town10HD'}
BIG={'Town11','Town12','Town13','Town15'}
def w(sel,p):
    n=ET.Element('routes'); [n.append(r) for r in sel]; ET.ElementTree(n).write(p,encoding='utf-8',xml_declaration=True)
w([r for r in root.findall('route') if r.get('town') in STD],"/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/B2D/leaderboard/data/full_eval.xml")
w([r for r in root.findall('route') if r.get('town') in BIG],"/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/B2D/leaderboard/data/large_eval.xml")
PY
}
[ -f leaderboard/data/full_eval.xml ] && [ -f leaderboard/data/large_eval.xml ] || build_routes
log "route: 표준 $(grep -c '<route ' leaderboard/data/full_eval.xml) / 대형 $(grep -c '<route ' leaderboard/data/large_eval.xml)"

# run_phase: rm 후 crash 재시도 루프로 target 채움. <backend> <routes> <outjson> <savepath> <logname>
run_phase(){
  local BK=$1 ROUTES=$2 OUT=$3 SAVE=$4 LOG=$5
  local TGT=$(grep -c '<route ' "$ROUTES")
  [ "$BK" = trt ] && [ "$SKIP_TRT" = 1 ] && { log "  $(basename $OUT) — SKIP_TRT=1 → 건너뜀"; return; }
  rm -f "$OUT"
  local stall=0
  while true; do
    local n=$(count "$OUT")
    [ "$n" -ge "$TGT" ] && { log "  $(basename $OUT) 완료 ($n/$TGT)"; break; }
    log "  $(basename $OUT) 진행 $n/$TGT — run_evaluation 시도 (stall=$stall)"
    kill_carla
    env E2E_BACKEND="$BK" \
      bash leaderboard/scripts/run_evaluation.sh 30000 50000 True "$ROUTES" "$AGENT" "$TEAM_CONFIG" \
      "$OUT" "$SAVE" "$PLANNER" 0 >> "leaderboard/data/$LOG" 2>&1
    local after=$(count "$OUT")
    [ "$after" -le "$n" ] && stall=$((stall+1)) || stall=0
    [ "$stall" -ge 3 ] && { log "  $(basename $OUT) 진척없음 3회 → 중단 ($after/$TGT)"; break; }
  done
}

scorecard(){ python3 DriveTransformer/scripts/scorecard.py "$1" "$2" > "$3" 2>&1; }

# ===== Phase A: 표준 55 =====
log "[Phase A] 표준 55 — PT"
run_phase pytorch leaderboard/data/full_eval.xml genad_full_pt.json  ./gd_full_pt  ovGA_pt.log
log "[Phase A] 표준 55 — TRT"
run_phase trt     leaderboard/data/full_eval.xml genad_full_trt.json ./gd_full_trt ovGA_trt.log
kill_carla
scorecard genad_full_pt.json genad_full_trt.json "$RES/SCORECARD.txt"
log "[Phase A] 채점표 → SCORECARD.txt"

# ===== Phase B: 대형 165 =====
log "[Phase B] 대형 165 — PT"
run_phase pytorch leaderboard/data/large_eval.xml genad_large_pt.json  ./gd_large_pt  ovGB_pt.log
log "[Phase B] 대형 165 — TRT"
run_phase trt     leaderboard/data/large_eval.xml genad_large_trt.json ./gd_large_trt ovGB_trt.log
kill_carla
scorecard genad_large_pt.json genad_large_trt.json "$RES/SCORECARD_LARGE.txt"
log "[Phase B] 채점표 → SCORECARD_LARGE.txt"

# ===== 전체 220 병합 채점표 =====
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
[ -f genad_all_pt.json ] && [ -f genad_all_trt.json ] && scorecard genad_all_pt.json genad_all_trt.json "$RES/SCORECARD_FULL.txt"
log "전체 220 채점표 → SCORECARD_FULL.txt"
log "ALL DONE — GenAD 220 PT vs TRT 완료"