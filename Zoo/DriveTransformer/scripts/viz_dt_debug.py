#!/usr/bin/env python3
# DT 회피 디버깅 viz — dump.npz 를 읽어 프레임별 top-down 오버레이(ego+탐지박스+궤적+waypoint) PNG/mp4 생성
import os
import argparse
import numpy as np
import cv2

# 9~10 클래스 이름/색 (BGR). 장애물성(barrier/cone)을 빨강 계열로.
NAMES = ["car", "truck", "constr_veh", "bus", "trailer", "barrier",
         "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
COLORS = {
    "car": (0, 200, 0), "truck": (0, 150, 0), "bus": (0, 150, 80), "trailer": (0, 120, 120),
    "constr_veh": (0, 90, 200), "barrier": (0, 0, 230), "traffic_cone": (0, 80, 255),
    "motorcycle": (200, 200, 0), "bicycle": (200, 160, 0), "pedestrian": (0, 230, 230),
}
CMD = {-1: "VOID", 0: "VOID", 1: "LEFT", 2: "RIGHT", 3: "STRAIGHT",
       4: "LANEFOLLOW", 5: "CL_L", 6: "CL_R"}


def _name(i):
    return NAMES[i] if 0 <= i < len(NAMES) else f"cls{i}"


class TopDown:
    """lidar 프레임(x=우, y=전방) → 픽셀. ego 는 화면 하단중앙, 전방이 위."""

    def __init__(self, size=760, range_m=36.0, fwd_bias=0.62):
        self.W = self.H = size
        self.R = range_m
        self.cx = size // 2
        self.cy = int(size * fwd_bias)   # ego 를 약간 아래에 둬 전방을 더 보여줌
        self.scale = (size / 2) / range_m

    def pt(self, x, y):
        return (int(self.cx + x * self.scale), int(self.cy - y * self.scale))

    def grid(self, img):
        for r in range(10, int(self.R) + 1, 10):
            cv2.circle(img, (self.cx, self.cy), int(r * self.scale), (60, 60, 60), 1)
            cv2.putText(img, f"{r}m", (self.cx + 3, self.cy - int(r * self.scale) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
        cv2.line(img, (self.cx, 0), (self.cx, self.H), (45, 45, 45), 1)
        cv2.line(img, (0, self.cy), (self.W, self.cy), (45, 45, 45), 1)


def draw_box(img, td, box, score, label):
    # box = [x,y,z, sx,sy,sz, yaw, vx,vy] (lidar). 수평 두 치수로 회전 사각형.
    x, y = float(box[0]), float(box[1])
    sx, sy = abs(float(box[3])), abs(float(box[4]))
    yaw = float(box[6])
    c, s = np.cos(yaw), np.sin(yaw)
    # 코너 (length=sx 를 heading 방향으로 가정 — 확인 후 조정)
    hx, hy = sx / 2, sy / 2
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    pts = []
    for dx, dy in corners:
        rx, ry = c * dx - s * dy, s * dx + c * dy
        pts.append(td.pt(x + rx, y + ry))
    color = COLORS.get(_name(label), (180, 180, 180))
    th = 2 if score > 0.5 else 1
    cv2.polylines(img, [np.array(pts, np.int32)], True, color, th)
    # heading 표시
    hx2, hy2 = c * hx - s * 0, s * hx + c * 0
    cv2.line(img, td.pt(x, y), td.pt(x + hx2, y + hy2), color, th)
    if score > 0.5:
        cv2.putText(img, f"{_name(label)[:3]}{score:.2f}", td.pt(x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)


def render(d, i, td, score_thr=0.3):
    img = np.full((td.H, td.W, 3), 22, np.uint8)
    td.grid(img)
    # ego (삼각형, 전방=위)
    e = td.pt(0, 0)
    cv2.drawContours(img, [np.array([(e[0], e[1] - 11), (e[0] - 8, e[1] + 8),
                                     (e[0] + 8, e[1] + 8)], np.int32)], 0, (255, 255, 255), -1)
    # 탐지박스
    sc, lb, bx = d["det_scores"][i], d["det_labels"][i], d["det_boxes"][i]
    for k in np.where(sc > score_thr)[0]:
        draw_box(img, td, bx[k], float(sc[k]), int(lb[k]))
    # 궤적: fix_time [30,2]=(fwd,lat) → 점(x=lat, y=fwd)
    ft = d["fix_time"][i]
    pts = [td.pt(p[1], p[0]) for p in ft]
    cv2.polylines(img, [np.array(pts, np.int32)], False, (255, 180, 0), 2)   # 주황 = fix_time
    for p in pts[::5]:
        cv2.circle(img, p, 2, (255, 180, 0), -1)
    # fix_dist [20,2]=(fwd,lat)
    fd = d["fix_dist"][i]
    pts2 = [td.pt(p[1], p[0]) for p in fd]
    cv2.polylines(img, [np.array(pts2, np.int32)], False, (255, 0, 200), 1)  # 보라 = fix_dist
    # waypoint near/far (ego-local)
    nl, fl = d["near_xy_local"][i], d["far_xy_local"][i]
    cv2.drawMarker(img, td.pt(nl[0], nl[1]), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 16, 2)  # near=노랑
    cv2.drawMarker(img, td.pt(fl[0], fl[1]), (255, 255, 0), cv2.MARKER_TILTED_CROSS, 16, 2)  # far=시안
    # 텍스트 HUD
    hud = [f"frame {i}  step {int(d['step'][i])}  t={d['timestamp'][i]:.1f}s",
           f"speed {d['speed'][i]:.1f} m/s   near={CMD.get(int(d['command_near'][i]))} far={CMD.get(int(d['command_far'][i]))}",
           f"det>{score_thr}: {int((sc>score_thr).sum())}   near_xy=({nl[0]:.1f},{nl[1]:.1f})",
           "orange=fix_time  magenta=fix_dist  Y-cross=near  C-cross=far"]
    for j, t in enumerate(hud):
        cv2.putText(img, t, (8, 20 + j * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", help="dump.npz 가 있는 폴더")
    ap.add_argument("--frame", type=int, default=None, help="단일 프레임만 렌더(미지정시 전체+mp4)")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--range", type=float, default=36.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--npz", default="dump.npz")
    args = ap.parse_args()

    d = np.load(os.path.join(args.dump_dir, args.npz))
    N = len(d["step"])
    td = TopDown(range_m=args.range)
    out_dir = os.path.join(args.dump_dir, "viz")
    os.makedirs(out_dir, exist_ok=True)

    if args.frame is not None:
        img = render(d, args.frame, td, args.score_thr)
        p = os.path.join(out_dir, f"frame_{args.frame:05d}.png")
        cv2.imwrite(p, img)
        print(f"saved {p}  ({N} frames total)")
        return

    vw = cv2.VideoWriter(os.path.join(args.dump_dir, "overlay.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (td.W, td.H))
    for i in range(N):
        img = render(d, i, td, args.score_thr)
        cv2.imwrite(os.path.join(out_dir, f"{i:05d}.png"), img)
        vw.write(img)
    vw.release()
    print(f"saved {N} PNG -> {out_dir}  + overlay.mp4")


if __name__ == "__main__":
    main()
