# DT TensorRT 추론 — backbone+head 엔진 직접 구동. PT model.forward_test 를 대체(agent 가 infer 호출).
from __future__ import annotations
import os
import numpy as np
import torch
import tensorrt as trt

_TRT2T = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
          trt.DataType.INT32: torch.int32}


class _Engine:
    """단일 TRT 엔진 wrapper — torch CUDA 텐서를 data_ptr 로 직결, 현재 stream 사용."""

    def __init__(self, path, device):
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        assert self.engine is not None, f"엔진 로드 실패: {path}"
        self.ctx = self.engine.create_execution_context()
        self.device = device
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            (self.inputs if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
             else self.outputs).append(n)

    def run(self, feed: dict) -> dict:
        keep = []
        for n in self.inputs:
            t = feed[n].contiguous().to(self.device, _TRT2T[self.engine.get_tensor_dtype(n)])
            keep.append(t)
            self.ctx.set_input_shape(n, tuple(t.shape))
            self.ctx.set_tensor_address(n, int(t.data_ptr()))
        out = {}
        for n in self.outputs:
            buf = torch.empty(tuple(self.ctx.get_tensor_shape(n)),
                              dtype=_TRT2T[self.engine.get_tensor_dtype(n)], device=self.device)
            out[n] = buf
            self.ctx.set_tensor_address(n, int(buf.data_ptr()))
        ok = self.ctx.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream)
        assert ok, "TRT execute_async_v3 실패"
        torch.cuda.current_stream(self.device).synchronize()
        return out


class DtTrt:
    def __init__(self, model_root=None, device="cuda:0"):
        root = model_root or os.environ.get(
            "B2D_MODEL", "/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/Model")
        base = os.path.join(root, "drivetransformer")
        self.device = torch.device(device)
        self.bb = _Engine(os.path.join(base, "backbone", "dt_backbone.engine"), self.device)
        self.head = _Engine(os.path.join(base, "head", "dt_head.engine"), self.device)
        self.token_location = self._build_token_location()
        self._dumped = False
        print(f"[DT-TRT] engines loaded | head inputs={len(self.head.inputs)} outputs={len(self.head.outputs)}")

    def _build_token_location(self):
        # head.prepare_location 재현: 12x33 grid, [ (w*32+16)/1056, (h*32+16)/384 ]
        H, W, s, padH, padW = 12, 33, 32, 384, 1056
        tl = np.zeros((H, W, 2), np.float32)
        for h in range(H):
            for w in range(W):
                tl[h, w, 0] = (w * s + s / 2) / padW
                tl[h, w, 1] = (h * s + s / 2) / padH
        return torch.from_numpy(tl).to(self.device)

    def smoke_test(self):
        img = torch.randn(6, 3, 384, 1056, device=self.device)
        feat = self.bb.run({self.bb.inputs[0]: img})[self.bb.outputs[0]]
        print("  [smoke] backbone:", tuple(feat.shape), "NaN:", bool(torch.isnan(feat).any()))
        return feat

    # ---------- memory (memory-off: 전부 0, egopose identity, prev_exists=0 → 단일프레임) ----------
    def _zero_memory(self):
        z = lambda *s: torch.zeros(s, dtype=torch.float32, device=self.device)
        ego = z(1, 10, 4, 4)
        for k in range(4):
            ego[:, :, k, k] = 1.0
        return {
            "prev_agent_memory_embedding": z(1, 500, 768),
            "prev_map_memory_embedding": z(1, 500, 768),
            "prev_ego_memory_embedding": z(1, 10, 768),
            "prev_agent_memory_class": z(1, 500, 9),
            "prev_agent_memory_reference_point": z(1, 500, 3),
            "prev_map_memory_reference_point": z(1, 500, 3),
            "prev_map_memory_class": z(1, 500, 6),
            "prev_memory_timestamp": z(1, 10, 1),
            "prev_memory_egopose": ego,
            "prev_memory_velo": z(1, 500, 2),
            "prev_memory_prev_exists": z(1, 10, 1),  # 0 → 모든 memory token mask out
        }

    # ---------- input_data_batch 추출 (DataContainer/list 중첩 풀기) ----------
    @staticmethod
    def _deep(x, depth=8):
        for _ in range(depth):
            if not torch.is_tensor(x) and hasattr(x, "data"):   # mmcv DataContainer
                x = x.data
            elif isinstance(x, (list, tuple)) and len(x) == 1:
                x = x[0]
            else:
                break
        return x

    def _dump(self, batch):
        print("[DT-TRT] === input_data_batch 구조 ===")
        for k, v in batch.items():
            d = self._deep(v)
            if torch.is_tensor(d):
                info = f"tensor{tuple(d.shape)} {d.dtype}"
            elif isinstance(d, dict):
                info = f"dict keys={list(d.keys())[:8]}"
            elif isinstance(d, np.ndarray):
                info = f"ndarray{d.shape}"
            else:
                info = f"{type(d).__name__} ({str(d)[:40]})"
            print(f"  {k:14s}: raw={type(v).__name__} -> {info}")

    def _tensor(self, batch, key, shape):
        x = self._deep(batch[key])
        x = torch.as_tensor(np.asarray(x) if not torch.is_tensor(x) else x,
                            dtype=torch.float32, device=self.device)
        return x.reshape(shape)

    @torch.no_grad()
    def infer(self, batch):
        if not self._dumped:
            self._dump(batch)
            self._dumped = True
        dev = self.device
        # --- img [6,3,384,1056] ---
        img = self._deep(batch["img"])
        img = img if torch.is_tensor(img) else torch.as_tensor(np.asarray(img))
        while img.dim() > 4:
            img = img[0]
        img = img.to(dev, torch.float32)
        # --- lidar2img / cam_intrinsic: top-level 키 (깨끗한 tensor [6,4,4]) ---
        l2i = torch.as_tensor(self._deep(batch["lidar2img"]), dtype=torch.float32, device=dev)      # [6,4,4]
        cami = torch.as_tensor(self._deep(batch["cam_intrinsic"]), dtype=torch.float32, device=dev)  # [6,4,4]
        # --- ego ---
        ego_lcf = self._tensor(batch, "ego_lcf_feat", (1, 1, 9))
        ego_cmd = self._tensor(batch, "ego_fut_cmd", (1, 1, 140))
        ego_his = self._tensor(batch, "ego_his_trajs", (1, 4))
        # --- backbone ---
        feat = self.bb.run({self.bb.inputs[0]: img})[self.bb.outputs[0]]   # [6,768,12,33]
        img_feats = feat.unsqueeze(0)                                       # [1,6,768,12,33]
        img2lidar = torch.inverse(l2i).unsqueeze(0)                         # [1,6,4,4]
        cam_intrinsic = cami[..., :3, :3].unsqueeze(0)                      # [1,6,3,3]
        feed = {
            "img_feats": img_feats, "img2lidar": img2lidar, "cam_intrinsic": cam_intrinsic,
            "ego_lcf_feat": ego_lcf, "ego_fut_cmd": ego_cmd, "ego_his_trajs": ego_his,
            "token_location": self.token_location,
        }
        feed.update(self._zero_memory())
        out = self.head.run(feed)
        # agent 가 쓰는 형식: output[0]['ego_fut_preds_fix_dist'][0,mode,:,0], ['ego_fut_preds_fix_time'][0,mode,:,[1,0]]
        return [{
            "ego_fut_preds_fix_time": out["ego_trajs"],                     # [1,1,30,2]
            "ego_fut_preds_fix_dist": out["ego_trajs_fix_dist"].unsqueeze(-1),  # [1,1,20,1]
        }]
