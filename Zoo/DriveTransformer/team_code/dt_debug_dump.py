# DT 회피 디버깅용 recorder — 프레임별 입력값(HLC/waypoint)·탐지(bbox)·궤적을 누적해 run당 단일 NPZ + CSV + viz 이미지로 저장
from __future__ import annotations
import os
import json
import time
import numpy as np

# B2D RoutePlanner RoadOption 정수 → 이름 (HLC 의미 해석용)
CMD_NAMES = {-1: "VOID", 0: "VOID", 1: "LEFT", 2: "RIGHT",
             3: "STRAIGHT", 4: "LANEFOLLOW", 5: "CHANGELANELEFT", 6: "CHANGELANERIGHT"}


def _np(x, dtype=np.float32):
    return np.ascontiguousarray(np.asarray(x, dtype=dtype))


class DebugRecorder:
    """run 동안 모든 프레임을 메모리에 누적했다가 save() 시 한 번에 떨군다.

    출력: <DT_DEBUG_DIR>/<backend>/<tag>/
      dump.npz   — 전 프레임 입력값 + 탐지(top-K 패딩) + 궤적을 프레임축으로 스택
      inputs.csv — 사람이 바로 읽는 프레임별 HLC/waypoint/speed 표
      meta.json  — 프레임 수 등 메타
      bev/NNNNN.png, cam_front/NNNNN.png — viz 배경/대조용 이미지

    env 제어 (DT_DEBUG_DIR 없으면 전부 no-op):
      DT_DEBUG_DIR        출력 루트
      DT_DEBUG_BACKEND    'Pytorch'|'ONNX'|'TensorRT' (기본 Pytorch)
      DT_DEBUG_TAG        run 이름 (기본 'run')
      DT_DEBUG_MAX        최대 프레임 수 (기본 무제한)
      DT_DEBUG_TOPK       탐지 패딩 K (기본 50)
      DT_DEBUG_SAVE_IMG   '1'이면 이미지 저장 (기본 1)
      DT_DEBUG_FRONT_EVERY  front 카메라 N프레임마다 저장 (기본 10)
    """

    def __init__(self, base_dir=None, backend=None, tag=None,
                 max_frames=None, topk=None, save_img=None, front_every=None):
        base_dir = base_dir if base_dir is not None else os.environ.get("DT_DEBUG_DIR", "")
        if not base_dir:
            self.enabled = False
            self.dir = None
            return
        backend = backend or os.environ.get("DT_DEBUG_BACKEND", "Pytorch")
        tag = tag or os.environ.get("DT_DEBUG_TAG", "") or "run"
        self.enabled = True
        self.topk = int(topk if topk is not None else os.environ.get("DT_DEBUG_TOPK", 50))
        _mx = max_frames if max_frames is not None else os.environ.get("DT_DEBUG_MAX", "")
        self.max_frames = int(_mx) if str(_mx) != "" else None
        self.save_img = str(save_img if save_img is not None
                            else os.environ.get("DT_DEBUG_SAVE_IMG", "1")) == "1"
        self.front_every = int(front_every if front_every is not None
                               else os.environ.get("DT_DEBUG_FRONT_EVERY", 10))
        self.dir = os.path.join(base_dir, backend, tag)
        os.makedirs(self.dir, exist_ok=True)
        if self.save_img:
            os.makedirs(os.path.join(self.dir, "bev"), exist_ok=True)
            os.makedirs(os.path.join(self.dir, "cam_front"), exist_ok=True)
        self._frames = []
        self._saved = False
        print(f"[DT-DBG] recorder enabled -> {self.dir} (topk={self.topk}, max={self.max_frames})")

    @classmethod
    def from_env(cls):
        return cls()

    def record(self, *, step, timestamp, command_near, command_far,
               near_xy_world, far_xy_world, near_xy_local, far_xy_local,
               speed, ego_lcf_feat, ego_pose, det, fix_time, fix_dist, angles,
               bev=None, cam_front=None):
        if not self.enabled:
            return
        if self.max_frames is not None and len(self._frames) >= self.max_frames:
            return
        boxes, scores, labels = self._extract_det(det)
        self._frames.append(dict(
            step=int(step), timestamp=float(timestamp),
            command_near=int(command_near), command_far=int(command_far),
            near_xy_world=_np(near_xy_world).reshape(-1)[:2],
            far_xy_world=_np(far_xy_world).reshape(-1)[:2],
            near_xy_local=_np(near_xy_local).reshape(-1)[:2],
            far_xy_local=_np(far_xy_local).reshape(-1)[:2],
            speed=float(speed), ego_lcf_feat=_np(ego_lcf_feat).reshape(-1),
            ego_pose=_np(ego_pose).reshape(4, 4),
            boxes=boxes, scores=scores, labels=labels,
            fix_time=_np(fix_time), fix_dist=_np(fix_dist), angles=_np(angles).reshape(-1),
        ))
        if self.save_img:
            self._save_img(int(step), len(self._frames) - 1, bev, cam_front)

    @staticmethod
    def _extract_det(det):
        empty = (np.zeros((0, 9), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64))
        if det is None:
            return empty
        try:
            b = det["boxes_3d"]
            b = b.tensor if hasattr(b, "tensor") else b
            boxes = np.asarray(b.detach().cpu().numpy(), np.float32)
            scores = np.asarray(det["scores_3d"].detach().cpu().numpy(), np.float32)
            labels = np.asarray(det["labels_3d"].detach().cpu().numpy(), np.int64)
            return boxes, scores, labels
        except Exception:
            return empty

    def _save_img(self, step, frame_idx, bev, cam_front):
        try:
            import cv2
            if bev is not None:
                cv2.imwrite(os.path.join(self.dir, "bev", f"{step:05d}.png"),
                            cv2.cvtColor(np.asarray(bev), cv2.COLOR_RGB2BGR))
            if cam_front is not None and (frame_idx % max(1, self.front_every) == 0):
                cv2.imwrite(os.path.join(self.dir, "cam_front", f"{step:05d}.png"),
                            cv2.cvtColor(np.asarray(cam_front), cv2.COLOR_RGB2BGR))
        except Exception:
            pass

    def save(self):
        if not self.enabled or self._saved or not self._frames:
            return
        self._saved = True
        F = len(self._frames)
        K = self.topk
        box_dim = max((f["boxes"].shape[1] for f in self._frames if f["boxes"].size), default=9)
        out = {
            "step": np.array([f["step"] for f in self._frames], np.int64),
            "timestamp": np.array([f["timestamp"] for f in self._frames], np.float32),
            "command_near": np.array([f["command_near"] for f in self._frames], np.int64),
            "command_far": np.array([f["command_far"] for f in self._frames], np.int64),
            "speed": np.array([f["speed"] for f in self._frames], np.float32),
        }
        for k in ("near_xy_world", "far_xy_world", "near_xy_local", "far_xy_local",
                  "ego_lcf_feat", "ego_pose", "fix_time", "fix_dist", "angles"):
            out[k] = np.stack([f[k] for f in self._frames]).astype(np.float32)
        # 탐지: top-K(score) 패딩
        det_boxes = np.zeros((F, K, box_dim), np.float32)
        det_scores = np.zeros((F, K), np.float32)
        det_labels = -np.ones((F, K), np.int64)
        det_count = np.zeros((F,), np.int64)
        for i, f in enumerate(self._frames):
            b, s, l = f["boxes"], f["scores"], f["labels"]
            det_count[i] = len(s)
            n = min(len(s), K)
            if n > 0:
                order = np.argsort(-s)[:n]
                det_boxes[i, :n, :b.shape[1]] = b[order]
                det_scores[i, :n] = s[order]
                det_labels[i, :n] = l[order]
        out.update(det_boxes=det_boxes, det_scores=det_scores,
                   det_labels=det_labels, det_count=det_count)
        npz_path = os.path.join(self.dir, "dump.npz")
        np.savez_compressed(npz_path, **out)
        self._write_csv()
        with open(os.path.join(self.dir, "meta.json"), "w") as fp:
            json.dump(dict(frames=F, topk=K, box_dim=int(box_dim),
                           dir=self.dir, saved_at=time.time()), fp, indent=2)
        print(f"[DT-DBG] saved {F} frames -> {npz_path}")

    def _write_csv(self):
        cols = ["step", "timestamp", "speed", "cmd_near", "cmd_far",
                "near_x_local", "near_y_local", "far_x_local", "far_y_local", "n_det"]
        lines = [",".join(cols)]
        for f in self._frames:
            nl, fl = f["near_xy_local"], f["far_xy_local"]
            lines.append(",".join(str(x) for x in [
                f["step"], round(f["timestamp"], 3), round(f["speed"], 3),
                CMD_NAMES.get(f["command_near"], f["command_near"]),
                CMD_NAMES.get(f["command_far"], f["command_far"]),
                round(float(nl[0]), 3), round(float(nl[1]), 3),
                round(float(fl[0]), 3), round(float(fl[1]), 3), int(f["scores"].shape[0])]))
        with open(os.path.join(self.dir, "inputs.csv"), "w") as fp:
            fp.write("\n".join(lines) + "\n")
