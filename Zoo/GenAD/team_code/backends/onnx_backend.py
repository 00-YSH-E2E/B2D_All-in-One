# ONNX Runtime 백엔드 — backbone는 ORT, head는 TRT 전용 plugin op 포함이라 PT fallback
from __future__ import annotations
from typing import Optional
import torch
import numpy as np

from .base import BackendBase, OUT_NAMES


# Head ONNX는 RotatePlugin 같은 TRT 전용 custom op를 포함하고 있어 onnxruntime
# 으로는 로드 불가. 본 백엔드는 backbone만 ORT로 돌리고 head는 fallback (PT)으로
# 위임한다. "ONNX 전 경로" 비교가 필요하면 head ONNX를 native op only로 재export
# 해야 함 (이 작업은 별건).


class OnnxBackend(BackendBase):
    """Hybrid: ORT-CUDA(or CPU) for backbone, PT fallback for head.

    Why head is PT: the exported head ONNX uses TRT custom ops (RotatePlugin
    and friends) that onnxruntime does not register, so it cannot load. To
    measure ORT vs TRT head, one would need to re-export the head without
    plugin ops — out of scope here.
    """

    name = "onnx"

    def __init__(self, backbone_onnx: str, head_onnx: Optional[str] = None,
                 head_fallback: Optional[BackendBase] = None, device: str = "cuda:0"):
        import onnxruntime as ort
        self._ort = ort
        self.device = device
        self.head_fallback = head_fallback  # set later by adapter if PT model available
        avail = set(ort.get_available_providers())
        providers = []
        if 'CUDAExecutionProvider' in avail:
            providers.append(('CUDAExecutionProvider', {'device_id': int(device.split(':')[-1])}))
        providers.append('CPUExecutionProvider')
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_bb = ort.InferenceSession(backbone_onnx, sess_opts, providers=providers)
        self._bb_in_name = self.sess_bb.get_inputs()[0].name
        self._bb_out_name = self.sess_bb.get_outputs()[0].name

        self.sess_head = None
        self._head_in_names = None
        self._head_out_names = None
        if head_onnx:
            try:
                self.sess_head = ort.InferenceSession(head_onnx, sess_opts, providers=providers)
                self._head_in_names = [i.name for i in self.sess_head.get_inputs()]
                self._head_out_names = [o.name for o in self.sess_head.get_outputs()]
            except Exception as e:
                print(f"[OnnxBackend] head_onnx failed to load ({e.__class__.__name__}): "
                      f"falling back to head_fallback. {e}")
                self.sess_head = None

    def _ort_value_from_tensor(self, t: torch.Tensor):
        t = t.contiguous()
        return self._ort.OrtValue.ortvalue_from_numpy(t.detach().cpu().numpy()) \
            if not t.is_cuda else self._ort.OrtValue.ortvalue_from_numpy(
                t.detach().cpu().numpy()  # fallback: CPU path
            )

    def _bind_and_run(self, sess, input_tensors: "dict[str, torch.Tensor]",
                      output_names: "list[str]") -> "list[torch.Tensor]":
        # Lightweight pure-numpy fallback path: copy in, sess.run, copy out.
        # IOBinding works in ORT 1.19 with torch CUDA pointers but the
        # exact API is fiddly across ORT versions — for the closed-loop
        # eval we prioritize correctness over per-frame perf.
        feeds = {}
        for name, t in input_tensors.items():
            feeds[name] = t.detach().cpu().numpy()
        outs = sess.run(output_names, feeds)
        return [torch.from_numpy(o).to(self.device) for o in outs]

    def run_backbone(self, img: torch.Tensor) -> torch.Tensor:
        # img is (B*N, 3, H, W) or (B, N, 3, H, W). The ONNX graph expects (6, 3, H, W).
        if img.dim() == 5:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
        else:
            B, N = 1, img.size(0)
        out, = self._bind_and_run(self.sess_bb, {self._bb_in_name: img}, [self._bb_out_name])
        # out shape: (B*N, 256, h, w) or (B, N, 256, h, w) depending on export.
        if out.dim() == 4:
            BN, C, h, w = out.shape
            out = out.view(B, BN // B, C, h, w)
        elif out.dim() == 5 and out.size(0) != B:
            out = out.view(B, -1, out.size(-3), out.size(-2), out.size(-1))
        return out

    def run_head(
        self,
        mlvl_feats_0: torch.Tensor,
        shift: torch.Tensor,
        lidar2img: torch.Tensor,
        can_bus: torch.Tensor,
        prev_bev: Optional[torch.Tensor],
    ) -> "dict[str, torch.Tensor]":
        if self.sess_head is None:
            # Plugin-op head ONNX unloadable in ORT → delegate to head_fallback.
            if self.head_fallback is None:
                raise RuntimeError(
                    "OnnxBackend.run_head: head ONNX could not load (TRT custom ops) "
                    "and no head_fallback configured. Either rebuild head ONNX with "
                    "native ops only, or set head_fallback=PyTorchBackend(model)."
                )
            return self.head_fallback.run_head(
                mlvl_feats_0, shift, lidar2img, can_bus, prev_bev
            )

        feeds = {
            'mlvl_feats.0': mlvl_feats_0,
            'img_metas.0[shift]': shift,
            'img_metas.0[lidar2img]': lidar2img,
            'img_metas.0[can_bus]': can_bus,
            'prev_bev': prev_bev,
        }
        feeds = {k: v for k, v in feeds.items() if k in self._head_in_names}
        out_names_ordered = ['out.' + n for n in OUT_NAMES]
        present = set(self._head_out_names)
        if not all(n in present for n in out_names_ordered):
            out_names_ordered = self._head_out_names
        outs = self._bind_and_run(self.sess_head, feeds, out_names_ordered)
        return dict(zip(OUT_NAMES, outs))

    def close(self) -> None:
        self.sess_bb = None
        self.sess_head = None
