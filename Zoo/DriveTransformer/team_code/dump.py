# GenAD step dumper — env GENAD_DUMP_DIR 설정 시 매 step 의 입력/출력 텐서를 NPZ 로 저장
"""Step-level NPZ dump utility for GenAD B2D inference.

Activated by env vars:
  GENAD_DUMP_DIR  : output directory. If unset/empty, all dumps are no-ops.
  GENAD_DUMP_TAG  : optional subdir tag, e.g. "town04_route0_onnx".
  GENAD_DUMP_MAX  : optional cap on number of steps dumped (default: 50).
                    Lets you grab the first N frames for inspection without
                    filling disk on a full route.

File naming: <dump_dir>/<tag>/step_NNNNNN.npz with named tensors.
Plus step_NNNNNN_meta.json with wall_time, step_index, scope label.

Two recommended call sites:
  1. Agent-level (vad_b2d_agent.run_step before model invocation):
       dumper.dump("agent_in", step, dict(img=..., can_bus_raw=..., ...))
  2. Adapter-level (inside patched extract_img_feat and head.forward):
       dumper.dump("backbone_in",  step, dict(img=...))
       dumper.dump("backbone_out", step, dict(features_l0=...))
       dumper.dump("head_in",      step, dict(mlvl_feats_0=..., shift=..., ...))
       dumper.dump("head_out",     step, dict(bev_embed=..., ego_fut_preds=..., ...))

Each scope label produces a separate NPZ; load with
  np.load("step_000005.npz") then dict-access by scope+tensor name.
"""
from __future__ import annotations
import json
import os
import time
from typing import Optional, Mapping, Any

import numpy as np


def _is_tensor(x):
    # Avoid importing torch at module load.
    return hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy")


def _to_numpy(x):
    if _is_tensor(x):
        x = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        pass
    elif isinstance(x, (list, tuple)):
        try:
            x = np.asarray(x)
        except Exception:
            return None
    elif isinstance(x, (int, float, bool)):
        x = np.asarray(x)
    else:
        return None
    return np.ascontiguousarray(x)


class StepDumper:
    def __init__(self, base_dir: Optional[str] = None, tag: Optional[str] = None,
                 max_steps: Optional[int] = None):
        base_dir = base_dir or os.environ.get("GENAD_DUMP_DIR", "")
        if not base_dir:
            self.enabled = False
            self.dir = None
            return
        tag = tag or os.environ.get("GENAD_DUMP_TAG", "default")
        try:
            max_env = int(os.environ.get("GENAD_DUMP_MAX", "50"))
        except ValueError:
            max_env = 50
        self.enabled = True
        self.max_steps = max_steps if max_steps is not None else max_env
        self.dir = os.path.join(base_dir, tag)
        os.makedirs(self.dir, exist_ok=True)
        # Per-scope payload accumulators for the current step.
        self._step_buf: dict = {}
        self._meta_buf: dict = {}

    @classmethod
    def from_env(cls):
        return cls()

    def _should_dump(self, step: int) -> bool:
        return self.enabled and (self.max_steps is None or step < self.max_steps)

    def dump(self, scope: str, step: int, tensors: Mapping[str, Any],
             meta: Optional[Mapping[str, Any]] = None) -> None:
        """Append `tensors` to the buffer for the given step under `scope`.
        Tensors are converted to numpy lazily. Call `flush(step)` to write the
        NPZ file when the step's collection is complete (or call `flush_now()`
        to write any partial buffer)."""
        if not self._should_dump(step):
            return
        for k, v in tensors.items():
            arr = _to_numpy(v)
            if arr is not None:
                self._step_buf[f"{scope}.{k}"] = arr
            else:
                # Non-tensor (e.g. dicts, list of ints) → stash in meta.
                self._meta_buf.setdefault(scope, {})[k] = _json_safe(v)
        if meta:
            self._meta_buf.setdefault(scope, {}).update({k: _json_safe(v) for k, v in meta.items()})

    def flush(self, step: int) -> None:
        if not self._should_dump(step):
            return
        if not self._step_buf and not self._meta_buf:
            return
        npz_path = os.path.join(self.dir, f"step_{step:06d}.npz")
        meta_path = os.path.join(self.dir, f"step_{step:06d}_meta.json")
        if self._step_buf:
            np.savez(npz_path, **self._step_buf)
        if self._meta_buf or True:
            self._meta_buf.setdefault("_", {})["wall_time"] = time.time()
            self._meta_buf["_"]["step"] = step
            with open(meta_path, "w") as f:
                json.dump(self._meta_buf, f, indent=2, default=str)
        self._step_buf = {}
        self._meta_buf = {}


def _json_safe(v):
    """Coerce metadata values into JSON-serializable form."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, np.ndarray):
        if v.size <= 32:
            return v.tolist()
        return {"_array_shape": list(v.shape), "_dtype": str(v.dtype)}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    return repr(v)
