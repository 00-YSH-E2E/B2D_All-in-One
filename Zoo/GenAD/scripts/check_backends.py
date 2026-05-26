#!/usr/bin/env python3
# CARLA 없이 PT/ONNX/TRT 백엔드 출력을 dummy 입력으로 수치 비교 — 어댑터 적분 검증용
"""Standalone sanity test for the 3 GenAD backends.

Usage (b2d_zoo env, Python 3.8):
    cd /home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/Zoo/GenAD
    python scripts/check_backends.py --modes pt,onnx
    python scripts/check_backends.py --modes pt,onnx,trt   # after engines built

Runs each requested backend through the same fixed (seeded) dummy inputs and
prints max/mean diff per output key. PT here applies the TRT-export patches so
its outputs match the engine's encoded semantics — that way diffs reflect
conversion fidelity, not patch semantics.
"""
import argparse
import os
import sys
import warnings

import numpy as np

# Ensure GenAD's vendored mmcv + adzoo wins on sys.path.
THIS = os.path.dirname(os.path.abspath(__file__))
GENAD_ROOT = os.path.abspath(os.path.join(THIS, '..'))
sys.path.insert(0, GENAD_ROOT)
sys.path.insert(0, os.path.join(GENAD_ROOT, 'adzoo/genad'))

# vad-trt patch source (needed only for PT-patched comparison).
VAD_TRT_EXPORT = '/home/Humble/Humble/Model_Optimization/DL4AGX/AV-Solutions/vad_base-trt/export_eval'
if os.path.isdir(VAD_TRT_EXPORT):
    sys.path.insert(0, VAD_TRT_EXPORT)

import torch

warnings.filterwarnings('ignore')

B2D_MODEL = os.environ.get('B2D_MODEL', '/home/Humble/Humble/EndtoEnd/Carla/Bench2Drive/Model')
CONFIG = os.path.join(GENAD_ROOT, 'adzoo/genad/configs/VAD/GenAD_config_b2d.py')
CKPT = os.path.join(B2D_MODEL, 'genad/pth/epoch_6.pth')

OUT_NAMES = [
    'bev_embed', 'all_cls_scores', 'all_bbox_preds',
    'all_traj_preds', 'all_traj_cls_scores',
    'map_all_cls_scores', 'map_all_bbox_preds',
    'map_all_pts_preds', 'ego_fut_preds',
]


def make_inputs(bev_h=100, bev_w=100, feat_h=23, feat_w=40,
                img_h=768, img_w=1280, embed=256):
    """Match verify_genad.py:make_inputs but parameterized."""
    np.random.seed(42)
    return {
        'img':         np.random.randn(6, 3, img_h, img_w).astype(np.float32),
        'mlvl_feats.0': np.random.randn(1, 6, embed, feat_h, feat_w).astype(np.float32),
        'shift':       np.zeros((1, 2), dtype=np.float32),
        'lidar2img':   np.tile(np.eye(4, dtype=np.float32), (6, 1, 1)),
        'can_bus':     np.zeros((1, 18), dtype=np.float32),
        'prev_bev':    np.random.randn(bev_h * bev_w, 1, embed).astype(np.float32),
    }


def _to_cuda(arr):
    return torch.from_numpy(arr).cuda()


def run_pt_patched(inputs, model):
    """Apply TRT-export patches then run backbone + head wrapper.
    This is the closest PT analog to what the engines compute."""
    from export_genad_head import apply_patches, GenADHeadWrapper

    print("[PT] applying TRT-export patches")
    apply_patches(model)
    wrapper = GenADHeadWrapper(model).cuda().eval()

    print("[PT] backbone forward")
    with torch.no_grad():
        bb_in = _to_cuda(inputs['img'])
        feats = model.img_backbone(bb_in)
        if isinstance(feats, dict):
            feats = list(feats.values())
        feats = model.img_neck(list(feats))
        bb_out = feats[0].detach()

    print("[PT] head forward (via GenADHeadWrapper)")
    with torch.no_grad():
        outs = wrapper(
            _to_cuda(inputs['mlvl_feats.0']),
            _to_cuda(inputs['shift']),
            _to_cuda(inputs['lidar2img']),
            _to_cuda(inputs['can_bus']),
            _to_cuda(inputs['prev_bev']),
        )
    head_out = dict(zip(OUT_NAMES, [o.detach() for o in outs]))
    return bb_out, head_out


def run_backend(backend, inputs):
    print(f"[{backend.name}] backbone")
    with torch.no_grad():
        bb_in = _to_cuda(inputs['img'])  # (6, 3, H, W) — adapter wraps for batch.
        bb_out = backend.run_backbone(bb_in)
        # Normalize shape to (6, C, h, w) for comparison.
        if bb_out.dim() == 5:
            B, N, C, h, w = bb_out.shape
            bb_out = bb_out.view(B * N, C, h, w)

    print(f"[{backend.name}] head")
    with torch.no_grad():
        head_out = backend.run_head(
            _to_cuda(inputs['mlvl_feats.0']),
            _to_cuda(inputs['shift']),
            _to_cuda(inputs['lidar2img']),
            _to_cuda(inputs['can_bus']),
            _to_cuda(inputs['prev_bev']),
        )
    return bb_out, head_out


def cmp_dict(label_a, label_b, bb_a, head_a, bb_b, head_b, fp16=False):
    print("\n" + "=" * 88)
    print(f"{label_a:>8} vs {label_b:<8}   " + "Output".ljust(30) + " max_diff      mean_diff")
    print("=" * 88)

    def _cmp(name, a, b, fp16_tol):
        if a.shape != b.shape:
            print(f"  {name:<30} SHAPE MISMATCH a={tuple(a.shape)} b={tuple(b.shape)}")
            return
        d = (a.float() - b.float()).abs()
        m = float(d.max().item())
        mn = float(d.mean().item())
        if fp16_tol:
            tag = "OK(fp16)" if m < 1.0 else ("CLOSE" if m < 5.0 else "SUSPECT")
        else:
            tag = "OK" if m < 1e-3 else ("CLOSE" if m < 0.1 else ("ACCUM" if m < 10.0 else "LARGE"))
        print(f"  {name:<30} {m:>12.4e}  {mn:>12.4e}  {tag}")

    _cmp('backbone', bb_a, bb_b, fp16)
    print("-" * 88)
    for n in OUT_NAMES:
        _cmp(n, head_a[n], head_b[n], False)
    print("=" * 88)


def build_pt_model():
    import mmcv
    from mmcv.utils import load_checkpoint
    from mmcv.models import build_model

    print(f"[setup] config: {CONFIG}")
    cfg = mmcv.Config.fromfile(CONFIG)
    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    print(f"[setup] ckpt:   {CKPT}")
    load_checkpoint(model, CKPT, map_location='cpu', strict=False)
    model.cuda().eval()
    return model


def bev_dims_from_config():
    """Read bev_h/bev_w from config without building the full model.
    Falls back to (100, 100) which is the GenAD B2D default."""
    try:
        import mmcv
        cfg = mmcv.Config.fromfile(CONFIG)
        head_cfg = cfg.model.get('pts_bbox_head', {})
        bev_h = head_cfg.get('bev_h', 100)
        bev_w = head_cfg.get('bev_w', 100)
        return int(bev_h), int(bev_w)
    except Exception as e:
        print(f"[setup] could not read bev dims from config ({e}); defaulting 100x100")
        return 100, 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', default='pt,onnx',
                    help='comma-separated subset of {pt,onnx,trt}')
    args = ap.parse_args()
    modes = [m.strip().lower() for m in args.modes.split(',') if m.strip()]
    print(f"[setup] modes: {modes}")
    print(f"[setup] B2D_MODEL: {B2D_MODEL}")

    pt_model = None
    if 'pt' in modes:
        pt_model = build_pt_model()
        bev_h, bev_w = pt_model.pts_bbox_head.bev_h, pt_model.pts_bbox_head.bev_w
    else:
        bev_h, bev_w = bev_dims_from_config()
    print(f"[setup] BEV: {bev_h} x {bev_w} = {bev_h*bev_w}")

    inputs = make_inputs(bev_h=bev_h, bev_w=bev_w)

    results = {}
    if 'pt' in modes:
        bb_out, head_out = run_pt_patched(inputs, pt_model)
        results['pt'] = (bb_out, head_out)

    if 'onnx' in modes:
        sys.path.insert(0, os.path.join(GENAD_ROOT, 'team_code'))
        from backends import OnnxBackend
        onnx_bb = os.path.join(B2D_MODEL, 'genad/backbone/genad_backbone.onnx')
        onnx_hd = os.path.join(B2D_MODEL, 'genad/head/genad_head.onnx')
        be = OnnxBackend(onnx_bb, onnx_hd)
        bb_out, head_out = run_backend(be, inputs)
        results['onnx'] = (bb_out, head_out)

    if 'trt' in modes:
        from backends import TrtBackend
        trt_bb = os.path.join(B2D_MODEL, 'genad/backbone/genad_backbone_fp16_trt86.engine')
        trt_hd = os.path.join(B2D_MODEL, 'genad/head/genad_head_fp32_trt86.engine')
        plugin = os.path.join(B2D_MODEL, 'genad/libplugins.so')
        if not os.path.exists(trt_bb) or not os.path.exists(trt_hd):
            print(f"[trt] engines not built yet. Skipping.\n  {trt_bb}\n  {trt_hd}")
        else:
            be = TrtBackend(trt_bb, trt_hd, plugin_lib=plugin if os.path.exists(plugin) else None)
            bb_out, head_out = run_backend(be, inputs)
            results['trt'] = (bb_out, head_out)

    # Pairwise comparisons.
    if 'pt' in results and 'onnx' in results:
        cmp_dict('pt', 'onnx', *results['pt'], *results['onnx'], fp16=False)
    if 'pt' in results and 'trt' in results:
        cmp_dict('pt', 'trt', *results['pt'], *results['trt'], fp16=True)
    if 'onnx' in results and 'trt' in results:
        cmp_dict('onnx', 'trt', *results['onnx'], *results['trt'], fp16=True)


if __name__ == '__main__':
    main()
