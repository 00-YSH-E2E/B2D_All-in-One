# 백엔드 ↔ mmcv 모델 연결 어댑터 — monkey-patch로 extract_img_feat / head.forward를 교체
from __future__ import annotations
import os
import types
import numpy as np
import torch

from .base import BackendBase, OUT_NAMES
from .pytorch_backend import PyTorchBackend
from .onnx_backend import OnnxBackend
from .trt_backend import TrtBackend


# Default search root for engines / ONNX. Overridden by env var B2D_MODEL.
_DEFAULT_B2D_MODEL = "/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/Model"


def _resolve_paths(model_kind: str = "genad"):
    root = os.environ.get("B2D_MODEL", _DEFAULT_B2D_MODEL)
    base = os.path.join(root, model_kind)
    return {
        "root": root,
        "backbone_onnx": os.path.join(base, "backbone", f"{model_kind}_backbone.onnx"),
        "head_onnx":     os.path.join(base, "head",     f"{model_kind}_head.onnx"),
        "backbone_trt":  os.path.join(base, "backbone", f"{model_kind}_backbone_fp16_trt86.engine"),
        "head_trt":      os.path.join(base, "head",     f"{model_kind}_head_fp32_trt86.engine"),
        "plugin_lib":    os.path.join(base, "libplugins.so"),
    }


def build_backend(name: str, model=None) -> BackendBase:
    name = name.lower().strip()
    if name in ("", "pt", "pytorch", "torch"):
        return PyTorchBackend(model)
    paths = _resolve_paths("genad")
    if name == "onnx":
        be = OnnxBackend(paths["backbone_onnx"], paths["head_onnx"])
        if be.sess_head is None and model is not None:
            be.head_fallback = PyTorchBackend(model)
        return be
    if name == "trt" or name == "tensorrt":
        plugin = paths["plugin_lib"] if os.path.exists(paths["plugin_lib"]) else None
        head_path = paths["head_trt"] if os.path.exists(paths["head_trt"]) else None
        be = TrtBackend(paths["backbone_trt"], head_path, plugin_lib=plugin)
        if be.head is None and model is not None:
            be.head_fallback = PyTorchBackend(model)
        return be
    raise ValueError(f"Unknown E2E_BACKEND: {name!r}")


def _compute_shift_from_can_bus(can_bus_np, ego_angle_deg, bev_h, bev_w,
                                grid_length=(0.512, 0.512), use_shift=True, device="cuda", dtype=torch.float32):
    """Replicate VAD_transformer.get_bev_features shift computation."""
    delta_x = float(can_bus_np[0])
    delta_y = float(can_bus_np[1])
    grid_y, grid_x = grid_length[0], grid_length[1]
    translation_length = float(np.sqrt(delta_x ** 2 + delta_y ** 2))
    translation_angle = float(np.arctan2(delta_y, delta_x) / np.pi * 180)
    bev_angle = ego_angle_deg - translation_angle
    shift_y = translation_length * float(np.cos(bev_angle / 180 * np.pi)) / grid_y / bev_h
    shift_x = translation_length * float(np.sin(bev_angle / 180 * np.pi)) / grid_x / bev_w
    if not use_shift:
        shift_x = 0.0
        shift_y = 0.0
    # shape: (bs=1, xy=2)
    return torch.tensor([[shift_x, shift_y]], device=device, dtype=dtype)


class BackendAdapter:
    """Installs/uninstalls monkey-patches that route the model's backbone and
    head through a BackendBase instance.

    Install policy:
      - PyTorchBackend → no-op (native forward retained).
      - OnnxBackend / TrtBackend → patch extract_img_feat and
        pts_bbox_head.forward.
    """

    def __init__(self, model, backend: BackendBase):
        self.model = model
        self.backend = backend
        self._orig_extract_img_feat = None
        self._orig_head_forward = None

    @classmethod
    def install(cls, model, backend: BackendBase) -> "BackendAdapter":
        self = cls(model, backend)
        if isinstance(backend, PyTorchBackend):
            return self  # native path
        self._patch_extract_img_feat()
        self._patch_head_forward()
        backend.warmup()
        return self

    def _patch_extract_img_feat(self):
        backend = self.backend
        self._orig_extract_img_feat = self.model.extract_img_feat

        def _patched_extract_img_feat(_self, img, img_metas, len_queue=None):
            if img is None:
                return None
            if img.dim() == 5 and img.size(0) == 1:
                img_in = img.squeeze(0)
                B, N = 1, img_in.size(0)
            elif img.dim() == 5:
                B, N, C, H, W = img.size()
                img_in = img.reshape(B * N, C, H, W)
            else:
                B, N = 1, img.size(0)
                img_in = img
            feat = backend.run_backbone(img_in)
            # Caller expects list[Tensor[B, N, C, h, w]]
            if feat.dim() == 4:
                BN, C, h, w = feat.shape
                feat = feat.view(B, BN // B, C, h, w)
            elif feat.dim() == 5 and feat.size(0) != B:
                feat = feat.view(B, -1, feat.size(-3), feat.size(-2), feat.size(-1))
            return [feat]

        self.model.extract_img_feat = types.MethodType(_patched_extract_img_feat, self.model)

    def _patch_head_forward(self):
        backend = self.backend
        head = self.model.pts_bbox_head
        self._orig_head_forward = head.forward
        # If the backend has a head fallback (e.g. OnnxBackend without working
        # head ONNX), give the fallback access to the unpatched forward so it
        # can call the native head without recursing into us.
        head_fallback = getattr(backend, 'head_fallback', None)
        if head_fallback is not None and hasattr(head_fallback, '_orig_head_forward'):
            head_fallback._orig_head_forward = self._orig_head_forward
        bev_h = head.bev_h
        bev_w = head.bev_w
        bev_len = bev_h * bev_w

        def _patched_head_forward(_self, mlvl_feats, img_metas, prev_bev=None,
                                  only_bev=False, **kwargs):
            assert not only_bev, "ONNX/TRT backend does not support only_bev=True"
            # mlvl_feats is list[Tensor]; engine takes the first level.
            mlvl0 = mlvl_feats[0] if isinstance(mlvl_feats, (list, tuple)) else mlvl_feats

            # img_metas comes as list[dict] (batched, bs=1 in B2D).
            meta = img_metas[0]
            can_bus_np = np.asarray(meta['can_bus'], dtype=np.float32)
            ego_angle_deg = float(can_bus_np[-2]) / float(np.pi) * 180.0
            device = mlvl0.device

            shift = _compute_shift_from_can_bus(
                can_bus_np, ego_angle_deg, bev_h, bev_w,
                grid_length=(0.512, 0.512),
                use_shift=getattr(_self.transformer, 'use_shift', True),
                device=device, dtype=mlvl0.dtype,
            )

            # lidar2img: (6, 4, 4)
            l2i = meta['lidar2img']
            if isinstance(l2i, (list, tuple)):
                l2i = np.stack([np.asarray(x, dtype=np.float32) for x in l2i], axis=0)
            else:
                l2i = np.asarray(l2i, dtype=np.float32)
            lidar2img = torch.from_numpy(l2i).to(device=device, dtype=mlvl0.dtype)

            # can_bus: (1, 18)
            can_bus = torch.from_numpy(can_bus_np[None, :]).to(device=device, dtype=mlvl0.dtype)

            # First-frame placeholder for prev_bev: zero [bev_len, 1, 256].
            if prev_bev is None:
                prev_bev_in = torch.zeros((bev_len, 1, mlvl0.size(2)), device=device, dtype=mlvl0.dtype)
            else:
                prev_bev_in = prev_bev

            outs = backend.run_head(mlvl0, shift, lidar2img, can_bus, prev_bev_in)
            # Sanity: all 9 keys present.
            missing = [k for k in OUT_NAMES if k not in outs]
            if missing:
                raise RuntimeError(f"backend missing keys {missing}; got {list(outs.keys())}")
            return outs

        head.forward = types.MethodType(_patched_head_forward, head)

    def uninstall(self):
        if self._orig_extract_img_feat is not None:
            self.model.extract_img_feat = self._orig_extract_img_feat
            self._orig_extract_img_feat = None
        if self._orig_head_forward is not None:
            self.model.pts_bbox_head.forward = self._orig_head_forward
            self._orig_head_forward = None

    def close(self):
        try:
            self.backend.close()
        finally:
            self.uninstall()
