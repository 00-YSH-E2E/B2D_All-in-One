#!/usr/bin/env python3
# DT B2D agent dump 의 input 텐서를 학습 dataset (B2D_DriveTransformer_Dataset) 의 formula와 대조
"""DT input mismatch checker.

Two modes:
  1. NPZ only (--npz)       : agent 의 dump 한 step 의 모든 입력을 출력 + 학습 formula 와 비교
                              (PKL 없이도 안 채워진 자리 / 부호 / 단위 mismatch 식별)
  2. NPZ + PKL (--pkl)      : 학습 데이터 한 sample 의 모든 입력도 같이 로드 후 라인 단위로 비교

Usage:
  python scripts/compare_dt_inputs.py \
    --npz /tmp/genad_dump/town04_dt_pytorch/step_000010.npz \
    --json /tmp/genad_dump/town04_dt_pytorch/step_000010_meta.json

  python scripts/compare_dt_inputs.py \
    --npz ... \
    --pkl /path/to/data/infos/per-route/route_xxx.pkl  --pkl-index 100
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


# 학습 dataset 의 can_bus 18-dim 구성 (B2D_DriveTransformer_Dataset.get_data_info 387~395)
CAN_BUS_LAYOUT = [
    ("[ 0]    x (m)",     "ego_translation[0]",                    "pos[0]",                       "x"),
    ("[ 1]    y (m)",     "ego_translation[1]",                    "-pos[1]  ⚠️ NEGATE",            "y"),
    ("[ 2]    z (m)",     "ego_translation[2]",                    "0       ⚠️ unset",             "z"),
    ("[ 3]  qx",          "rotation[0]",                           "rotation[0]",                  "ok"),
    ("[ 4]  qy",          "rotation[1]",                           "rotation[1]",                  "ok"),
    ("[ 5]  qz",          "rotation[2]",                           "rotation[2]",                  "ok"),
    ("[ 6]  qw",          "rotation[3]",                           "rotation[3]",                  "ok"),
    ("[ 7]   vx (m/s)",   "ego_vel[0]",                            "tick['speed']",                "ok"),
    ("[ 8]   vy (m/s)",   "ego_vel[1]",                            "0       ⚠️ unset",             "ok"),
    ("[ 9]   vz (m/s)",   "ego_vel[2]",                            "0       ⚠️ unset",             "ok"),
    ("[10]   ax (m/s²)",  "ego_accel[0]",                          "accel[0]",                     "ok"),
    ("[11]   ay (m/s²)",  "ego_accel[1]",                          "accel[1] * -1  ⚠️ NEGATE",     "ok"),
    ("[12]   az (m/s²)",  "ego_accel[2]",                          "accel[2]",                     "ok"),
    ("[13]   ωx (rad/s)", "ego_rotation_rate[0]",                  "-ω[0]   ⚠️ NEGATE",            "ok"),
    ("[14]   ωy (rad/s)", "ego_rotation_rate[1]",                  "-ω[1]   ⚠️ NEGATE",            "ok"),
    ("[15]   ωz (rad/s)", "ego_rotation_rate[2]",                  "-ω[2]   ⚠️ NEGATE",            "ok"),
    ("[16] yaw (rad)",    "yaw",                                   "ego_theta",                    "ok"),
    ("[17] yaw (deg)",    "yaw / pi * 180",                        "ego_theta / pi * 180",         "ok"),
]


def load_agent(npz_path: str, json_path: str = None):
    """Load one step's NPZ + JSON sidecar. Returns dict-like with both."""
    out = {}
    if npz_path and os.path.exists(npz_path):
        z = np.load(npz_path)
        out["npz"] = {k: z[k] for k in z.files}
    else:
        out["npz"] = {}
    json_path = json_path or (npz_path.replace(".npz", "_meta.json") if npz_path else None)
    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            out["json"] = json.load(f)
    else:
        out["json"] = {}
    return out


def load_b2d_pkl(pkl_path: str, idx: int = 0):
    """Load one sample from a B2D infos PKL.

    Format (per per-route pkl): list[dict], each dict has keys like
    'ego_translation', 'ego_vel', 'ego_accel', 'ego_rotation_rate', 'cams', etc.
    """
    import pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "infos" in data:
        data = data["infos"]
    if not isinstance(data, list):
        raise ValueError(f"unexpected pkl structure: {type(data)}")
    if idx >= len(data):
        raise IndexError(f"pkl has {len(data)} samples, idx={idx} OOR")
    return data[idx]


def show_can_bus(agent, b2d_sample=None):
    print("\n" + "=" * 110)
    print("CAN_BUS (18-dim)  —  training dataset.py:387-395  vs  agent.py:407-417")
    print("=" * 110)
    agent_cb = agent["npz"].get("agent_in.can_bus")
    if agent_cb is None:
        agent_cb = agent["json"].get("agent_in", {}).get("can_bus")
        if agent_cb is not None:
            agent_cb = np.asarray(agent_cb, dtype=np.float64)
    if agent_cb is None:
        print("  ❌ agent dump 에 can_bus 없음.")
        return
    agent_cb = np.asarray(agent_cb).reshape(-1)
    if agent_cb.size != 18:
        print(f"  ⚠️ can_bus 크기가 18 아님: {agent_cb.size}")

    # Build a "training side" reconstruction if PKL provided.
    train_cb = None
    if b2d_sample is not None:
        train_cb = np.zeros(18, dtype=np.float64)
        try:
            train_cb[0:3]   = b2d_sample.get('ego_translation', [0]*3)
            r = b2d_sample.get('ego_rotation') or b2d_sample.get('rotation')
            if r is not None:
                train_cb[3:7] = r
            train_cb[7:10]  = b2d_sample.get('ego_vel', [0]*3)
            train_cb[10:13] = b2d_sample.get('ego_accel', [0]*3)
            train_cb[13:16] = b2d_sample.get('ego_rotation_rate', [0]*3)
            yaw = b2d_sample.get('ego_yaw')
            if yaw is None:
                # try compute from quaternion: yaw = atan2(2(qw·qz + qx·qy), 1 - 2(qy² + qz²))
                if r is not None:
                    qx, qy, qz, qw = float(r[0]), float(r[1]), float(r[2]), float(r[3])
                    yaw = float(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
                    if yaw < 0: yaw += 2*np.pi
                else:
                    yaw = 0.0
            train_cb[16] = yaw
            train_cb[17] = yaw / np.pi * 180
        except Exception as e:
            print(f"  ⚠️ 학습 sample reconstruct 실패: {e}")
            train_cb = None

    header = f"  {'idx | desc':<22} | {'training formula':<32} | {'agent formula':<28} | "
    if train_cb is not None:
        header += f"{'train_value':>12} | {'agent_value':>12} | diff"
    else:
        header += f"{'agent_value':>12}"
    print(header)
    print("  " + "-" * (len(header)-2))
    for i, (desc, train_form, agent_form, _) in enumerate(CAN_BUS_LAYOUT):
        av = float(agent_cb[i])
        line = f"  {desc:<22} | {train_form:<32} | {agent_form:<28} | "
        if train_cb is not None:
            tv = float(train_cb[i])
            d = av - tv
            tag = ""
            if abs(d) > 1e-3:
                tag = "  ⚠️ diff"
            line += f"{tv:>12.4f} | {av:>12.4f} | {d:>+10.4f}{tag}"
        else:
            line += f"{av:>12.4f}"
        print(line)


def show_ego_lcf(agent, b2d_sample=None):
    print("\n" + "=" * 90)
    print("EGO_LCF_FEAT (9-dim)  —  dataset.py:398-405  vs  agent.py:420-427")
    print("=" * 90)
    v = agent["npz"].get("agent_in.ego_lcf_feat")
    if v is None:
        print("  ❌ agent dump 에 ego_lcf_feat 없음.")
        return
    v = np.asarray(v).reshape(-1)
    layout = [
        ("[0] vx",          "ego_vel[0]",          "tick['speed']"),
        ("[1] vy",          "ego_vel[1]",          "0  ⚠️ unset"),
        ("[2] ax",          "ego_accel[2]?",       "can_bus_acc[10]"),
        ("[3] ay",          "ego_accel[3]?",       "can_bus_acc[11]"),
        ("[4] yaw_rate",    "ego_rotation_rate[-1]","can_bus_ω[15]"),
        ("[5] ego_size[1]", "차폭 (m)",            "1.83671331 (hardcoded)"),
        ("[6] ego_size[0]", "차장 (m)",            "4.89238167 (hardcoded)"),
        ("[7] vx dup",      "ego_vel[0]",          "tick['speed']"),
        ("[8] prev_steer",  "steer",               "prev_control_cache[0].steer"),
    ]
    for i, (desc, tf, af) in enumerate(layout):
        av = float(v[i]) if i < v.size else float("nan")
        print(f"  {desc:<18} | {tf:<25} | {af:<32} | agent={av:>10.4f}")


def show_ego_his_trajs(agent, b2d_sample=None):
    print("\n" + "=" * 80)
    print("EGO_HIS_TRAJS (shape [2,2])  —  dataset.py:413,631  vs  agent.py:454-462")
    print("=" * 80)
    v = agent["npz"].get("agent_in.ego_his_trajs")
    if v is None:
        print("  ❌ agent dump 에 ego_his_trajs 없음.")
        return
    v = np.asarray(v)
    print(f"  agent shape={v.shape}")
    print(f"  agent values:\n{v}")
    print()
    print("  training: sample_interval=5, past_frames=2 → 2Hz 데이터에서 5/10 frame 차이 (=0.5s,1s)")
    print("  agent:    20Hz cache 에서 [-10]·[0] 인덱스 (=0.5s, 1s)")
    print("  → 단위 시간은 같지만 값 의미가 다를 수 있음 (cache[-10] = 절대 pose vs training = offset_track)")


def show_ego_fut_cmd(agent):
    print("\n" + "=" * 80)
    print("EGO_FUT_CMD (140-dim)  —  one-hot + sinusoidal pos embedding")
    print("=" * 80)
    v = agent["npz"].get("agent_in.ego_fut_cmd")
    if v is None:
        print("  ❌ agent dump 에 ego_fut_cmd 없음.")
        return
    v = np.asarray(v).reshape(-1)
    far_oh = v[0:6]
    near_oh = v[70:76]
    far_emb = v[6:70]
    near_emb = v[76:140]
    print(f"  far_command one-hot  [0:6]   : {far_oh.tolist()}  → class {int(np.argmax(far_oh))}")
    print(f"  far_command emb  norm[6:70]  : ||·||={np.linalg.norm(far_emb):.3f}  (sin/cos 64-dim)")
    print(f"  near_command one-hot [70:76] : {near_oh.tolist()} → class {int(np.argmax(near_oh))}")
    print(f"  near_command emb norm[76:140]: ||·||={np.linalg.norm(near_emb):.3f}")
    print("  ⚠️ theta_to_lidar = raw_theta (agent line 433) vs training의 -ego_theta + pi/2 (dataset line 435)")


def show_lidar2img(agent):
    print("\n" + "=" * 80)
    print("LIDAR2IMG (shape [6,4,4])  —  6 cameras × 4×4 matrix")
    print("=" * 80)
    v = agent["json"].get("agent_in", {}).get("lidar2img")
    if v is None:
        v = agent["npz"].get("agent_in.lidar2img")
        if v is not None:
            v = v.tolist()
    if v is None:
        print("  ❌ lidar2img 없음 (npz/json 둘 다).")
        return
    v = np.asarray(v)
    print(f"  shape: {v.shape}")
    # K = first 3×3 block
    cams = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "BACK", "BACK_LEFT", "BACK_RIGHT"]
    print(f"  K (fx,fy,cx,cy) per camera:")
    for i, name in enumerate(cams):
        if i >= v.shape[0]: break
        M = v[i]
        # M = K_pad @ E ; fx ≈ M[0,0] / scale 의 정확한 분해는 어렵지만 (fx, fy) 위치 출력
        print(f"    {name:>11}: M[0,0]={M[0,0]:>9.2f}  M[1,1]={M[1,1]:>9.2f}  M[0,2]={M[0,2]:>9.2f}  M[1,2]={M[1,2]:>9.2f}")


def show_img_stats(agent):
    print("\n" + "=" * 80)
    print("IMG (정규화 후 통계)  —  training: mean=[123.7,116.3,103.5], std=[58.4,57.1,57.4]")
    print("=" * 80)
    img = agent["npz"].get("agent_in.img")
    if img is None:
        print("  ❌ agent dump 에 img 없음.")
        return
    img = np.asarray(img)
    print(f"  shape: {img.shape}  dtype: {img.dtype}")
    if img.ndim >= 3:
        flat = img.reshape(-1)
        print(f"  global mean: {flat.mean():.4f}  std: {flat.std():.4f}  min: {flat.min():.4f}  max: {flat.max():.4f}")
        # If properly normalized (subtract mean / divide std), centered ~0, std ~1.
        # If NOT normalized (raw 0-255), mean 가 100 근처.
        if abs(flat.mean()) < 5 and 0.5 < flat.std() < 2.5:
            print("  ✅ 정규화된 분포 (centered, std~1) — pipeline의 NormalizeMultiviewImage 가 적용된 듯")
        elif 50 < flat.mean() < 200:
            print("  ⚠️ 정규화 안 됨 — raw uint8 그대로? 학습은 정규화된 입력 기대")
        else:
            print("  ⚠️ 분포가 예상 범위 밖. 확인 필요")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="agent dump step_NNNNNN.npz 경로")
    ap.add_argument("--json", default=None, help="agent dump step_NNNNNN_meta.json (기본: npz 경로 기반 자동)")
    ap.add_argument("--pkl", default=None, help="(선택) B2D 학습 sample PKL — 라인별 수치 비교")
    ap.add_argument("--pkl-index", type=int, default=0, help="PKL 안에서 sample 인덱스")
    args = ap.parse_args()

    print(f"[load] NPZ : {args.npz}")
    agent = load_agent(args.npz, args.json)
    npz_keys = sorted(agent["npz"].keys())
    print(f"[load] NPZ keys ({len(npz_keys)}): {npz_keys}")
    json_keys = sorted((agent["json"].get("agent_in") or {}).keys())
    print(f"[load] JSON agent_in keys: {json_keys}")

    b2d = None
    if args.pkl:
        print(f"[load] PKL : {args.pkl} (idx={args.pkl_index})")
        try:
            b2d = load_b2d_pkl(args.pkl, args.pkl_index)
            print(f"[load] PKL sample keys: {sorted(list(b2d.keys()))[:15]}{'...' if len(b2d)>15 else ''}")
        except Exception as e:
            print(f"[load] PKL load 실패: {e}")

    show_can_bus(agent, b2d)
    show_ego_lcf(agent, b2d)
    show_ego_his_trajs(agent, b2d)
    show_ego_fut_cmd(agent)
    show_lidar2img(agent)
    show_img_stats(agent)

    print("\n" + "=" * 80)
    print("결론 요약")
    print("=" * 80)
    print("  ⚠️ 마크 보이면 → 학습 input 분포와 다를 가능성. PT eval 점수가 낮으면 여기부터 fix.")
    print("  ✅ 마크 보이면 → 그 필드는 학습 formula 와 매칭.")
    print()
    if b2d is None:
        print("  📌 PKL 없이 실행 — agent 값만 표시 + 학습 formula 비교 (수치 diff 없음).")
        print("     학습 PKL 한 sample 가져오면 --pkl 로 넘겨서 정확한 라인 diff 가능.")


if __name__ == "__main__":
    main()
