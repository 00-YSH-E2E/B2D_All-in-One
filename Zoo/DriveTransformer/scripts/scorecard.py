#!/usr/bin/env python3
# PT vs TRT 채점표 — 두 leaderboard result.json 을 route_id 로 매칭해 시나리오별 비교 + 요약
import json
import sys
import statistics as st


def load(p):
    out = {}
    try:
        d = json.load(open(p))
    except Exception as e:
        print(f"(load 실패 {p}: {e})")
        return out
    for r in d["_checkpoint"]["records"]:
        inf = r["infractions"]
        out[r["route_id"]] = dict(
            scn=r["scenario_name"], status=r["status"],
            route=r["scores"]["score_route"], comp=r["scores"]["score_composed"],
            cv=len(inf.get("collisions_vehicle", [])),
            cl=len(inf.get("collisions_layout", [])),
            cp=len(inf.get("collisions_pedestrian", [])),
            ms=len(inf.get("min_speed_infractions", [])),
        )
    return out


def main():
    pt = load(sys.argv[1])
    trt = load(sys.argv[2])
    keys = sorted(set(pt) | set(trt))
    print(f"=== PT vs TRT 채점표 (PT {len(pt)}개 / TRT {len(trt)}개 완료) ===\n")
    hdr = f"{'route':>22} {'scenario':26} |{'PTcomp':>7}{'cv':>3}{'cl':>3} |{'TRTcomp':>8}{'cv':>3}{'cl':>3} | 회피일치"
    print(hdr)
    print("-" * len(hdr))
    ptc, trtc, match, both = [], [], 0, 0
    for k in keys:
        p, t = pt.get(k), trt.get(k)
        ps = f"{p['comp']:7.0f}{p['cv']:3d}{p['cl']:3d}" if p else f"{'-':>7}{'-':>3}{'-':>3}"
        ts = f"{t['comp']:8.0f}{t['cv']:3d}{t['cl']:3d}" if t else f"{'-':>8}{'-':>3}{'-':>3}"
        scn = (p or t)["scn"][:26]
        m = ""
        if p and t:
            both += 1
            ptc.append(p["comp"]); trtc.append(t["comp"])
            same = (p["cv"] == 0) == (t["cv"] == 0)
            if same:
                match += 1
                m = "O"
            else:
                m = f"X (PT cv{p['cv']} / TRT cv{t['cv']})"
        print(f"{k[:22]:>22} {scn:26} |{ps} |{ts} | {m}")
    print("\n=== 요약 ===")
    print(f"  둘 다 완료한 route: {both}")
    if ptc:
        print(f"  PT  평균 driving score(composed): {st.mean(ptc):.1f}")
    if trtc:
        print(f"  TRT 평균 driving score(composed): {st.mean(trtc):.1f}")
    if both:
        print(f"  차량회피 일치(둘 다 충돌0 또는 둘 다 충돌): {match}/{both} ({100*match/both:.0f}%)")


if __name__ == "__main__":
    main()
