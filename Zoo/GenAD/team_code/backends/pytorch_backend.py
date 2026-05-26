# PyTorch 레퍼런스 백엔드 — 원본 mmcv 모듈에 그대로 위임 (패치 없이 as-trained 동작)
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from .base import BackendBase, OUT_NAMES


class PyTorchBackend(BackendBase):
    """Delegates to the live PyTorch model — used as the B2D baseline.

    Important: we do NOT apply the TRT-export patches here. The agent should
    drive with the as-trained mmcv model. If you want patched-PT for
    numerical comparison, use the apply_patches path inside
    scripts/check_backends.py — not here.
    """

    name = "pytorch"

    def __init__(self, model: nn.Module):
        self.model = model
        # Bound to the live head — when used as a fallback under a monkey-
        # patched head, callers must override this with the saved original
        # head.forward to avoid recursing into the patch.
        self._orig_head_forward = None

    def run_backbone(self, img: torch.Tensor) -> torch.Tensor:
        # Mirror GenAD.extract_img_feat lines 64-97 but return the first level
        # already shaped (B, N, C, h, w). mmcv's extract_img_feat will be
        # monkey-patched to call us; we replicate its reshape contract.
        m = self.model
        if img.dim() == 5 and img.size(0) == 1:
            img = img.squeeze(0)
            B = 1
            N = img.size(0)
        elif img.dim() == 5:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
        else:
            BN = img.size(0)
            # Assume B=1 (CARLA driving — single-batch). N inferred later.
            B = 1
            N = BN
        feats = m.img_backbone(img)
        if isinstance(feats, dict):
            feats = list(feats.values())
        if m.with_img_neck:
            feats = m.img_neck(feats)
        # Return first level reshaped to (B, N, C, h, w) — matches what the
        # patched extract_img_feat[0] would produce.
        f0 = feats[0]
        BNc, C, h, w = f0.size()
        return f0.view(B, BNc // B, C, h, w)

    def run_head(
        self,
        mlvl_feats_0: torch.Tensor,
        shift: torch.Tensor,
        lidar2img: torch.Tensor,
        can_bus: torch.Tensor,
        prev_bev: Optional[torch.Tensor],
    ) -> "dict[str, torch.Tensor]":
        # The PT backend's head path is normally NOT routed through here —
        # the adapter installs run_head only for ONNX/TRT, and PT keeps
        # native forward. Provided for symmetry / check_backends.py / as
        # ONNX fallback (when head ONNX is plugin-only and can't load).
        img_metas = [{
            'shift': shift,
            'lidar2img': lidar2img,
            'can_bus': can_bus,
            'img_shape': [(768, 1280, 3)] * 6,
        }]
        head = self.model.pts_bbox_head
        if self._orig_head_forward is not None:
            # Use the saved original to avoid recursing into a monkey-patched forward.
            outs = self._orig_head_forward(
                [mlvl_feats_0], img_metas, prev_bev=prev_bev, only_bev=False
            )
        else:
            outs = head([mlvl_feats_0], img_metas, prev_bev=prev_bev, only_bev=False)
        return {k: outs[k] for k in OUT_NAMES}
