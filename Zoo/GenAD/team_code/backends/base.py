# GenAD 백엔드 공통 인터페이스 — backbone/head 추론을 PT/ONNX/TRT가 동일 시그니처로 제공
from __future__ import annotations
from typing import Optional
import torch

OUT_NAMES = [
    'bev_embed',
    'all_cls_scores',
    'all_bbox_preds',
    'all_traj_preds',
    'all_traj_cls_scores',
    'map_all_cls_scores',
    'map_all_bbox_preds',
    'map_all_pts_preds',
    'ego_fut_preds',
]


class BackendBase:
    """Backend interface for swapping the GenAD net while keeping mmcv pipeline.

    Contract:
      - run_backbone receives img on CUDA shape [B*N, 3, H, W] (mmcv flattens
        N cams into batch axis); returns the first-level mlvl feature with
        shape [B, N, 256, h, w] (matches mmcv's view-reshape after extract).
      - run_head receives the 5 engine-compatible tensors and returns a dict
        keyed by OUT_NAMES; every value is a torch.Tensor on CUDA. mmcv
        downstream (`get_bboxes`, `forward_test`) consumes the dict as if it
        came from native GenADHead.forward.
    """

    name = "base"

    def run_backbone(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def run_head(
        self,
        mlvl_feats_0: torch.Tensor,
        shift: torch.Tensor,
        lidar2img: torch.Tensor,
        can_bus: torch.Tensor,
        prev_bev: Optional[torch.Tensor],
    ) -> "dict[str, torch.Tensor]":
        raise NotImplementedError

    def warmup(self) -> None:
        pass

    def close(self) -> None:
        pass
