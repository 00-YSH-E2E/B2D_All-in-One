# TensorRT 백엔드 — torch CUDA 스트림을 공유하고 data_ptr()로 IO 직결 (pycuda 미사용)
from __future__ import annotations
from typing import Optional
import os
import ctypes
import torch

from .base import BackendBase, OUT_NAMES


def _import_trt():
    import tensorrt as trt  # local import to keep PT-only paths cheap
    return trt


_NP_TO_TORCH = None


def _trt_dtype_to_torch(trt_dtype):
    trt = _import_trt()
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
    }
    if hasattr(trt.DataType, 'INT64'):
        mapping[trt.DataType.INT64] = torch.int64
    if hasattr(trt.DataType, 'BOOL'):
        mapping[trt.DataType.BOOL] = torch.bool
    return mapping[trt_dtype]


class _TrtRunner:
    """Single-engine wrapper. Holds context + persistent IO buffers (torch CUDA).

    Buffer allocation policy:
      - Inputs: bind directly to caller-provided tensors via data_ptr().
      - Outputs: pre-allocated as torch CUDA tensors at engine load time
        (shapes are static for our two engines — backbone takes fixed 6x3xHxW
        and head takes fixed shapes per verify_genad.py).
    """

    def __init__(self, engine_path: str, device: torch.device):
        trt = _import_trt()
        self._trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, '')
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"TRT engine deserialization failed: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        self.device = device

        # Enumerate IO tensors.
        self.input_names: list = []
        self.output_names: list = []
        self.output_buffers: "dict[str, torch.Tensor]" = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
                if any(d < 0 for d in shape):
                    # Dynamic output — defer allocation to first call.
                    self.output_buffers[name] = None  # type: ignore
                else:
                    self.output_buffers[name] = torch.empty(
                        shape, dtype=dtype, device=device
                    )
                self.ctx.set_tensor_address(
                    name,
                    int(self.output_buffers[name].data_ptr()) if self.output_buffers[name] is not None else 0,
                )

    def run(self, inputs: "dict[str, torch.Tensor]") -> "dict[str, torch.Tensor]":
        trt = self._trt
        for name in self.input_names:
            t = inputs.get(name)
            if t is None:
                raise KeyError(f"Engine input '{name}' not provided. Have: {list(inputs.keys())}")
            t = t.contiguous().to(self.device)
            # Match dtype to engine expectation.
            expected_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            if t.dtype != expected_dtype:
                t = t.to(expected_dtype)
            inputs[name] = t  # keep alive for the call
            self.ctx.set_input_shape(name, tuple(t.shape))
            self.ctx.set_tensor_address(name, int(t.data_ptr()))

        # Re-check / re-alloc dynamic outputs based on input-shape-set context.
        for name in self.output_names:
            shape = tuple(self.ctx.get_tensor_shape(name))
            buf = self.output_buffers.get(name)
            if buf is None or tuple(buf.shape) != shape:
                dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
                buf = torch.empty(shape, dtype=dtype, device=self.device)
                self.output_buffers[name] = buf
                self.ctx.set_tensor_address(name, int(buf.data_ptr()))

        stream_handle = torch.cuda.current_stream(self.device).cuda_stream
        ok = self.ctx.execute_async_v3(stream_handle)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 returned False")
        torch.cuda.current_stream(self.device).synchronize()
        return {n: self.output_buffers[n] for n in self.output_names}


class TrtBackend(BackendBase):
    """TensorRT swap-in.

    backbone_engine is always required. head_engine is optional — if absent
    (or fails to load), run_head delegates to head_fallback (typically a
    PyTorchBackend). This accommodates the known TRT 8.6 ONNX-parser limit
    on the GenAD head ONNX (dynamic Slice axes + If conditional shape
    mismatch). To run a full TRT head path, either re-export the head ONNX
    without those constructs or use TRT 10+ trtexec.
    """

    name = "trt"

    def __init__(self, backbone_engine: str, head_engine: Optional[str] = None,
                 plugin_lib: Optional[str] = None,
                 head_fallback: Optional[BackendBase] = None,
                 device: str = "cuda:0"):
        self.device = torch.device(device)
        self.head_fallback = head_fallback
        # Load plugin lib globally BEFORE creating any TRT runtime.
        if plugin_lib:
            if not os.path.exists(plugin_lib):
                raise FileNotFoundError(f"plugin_lib not found: {plugin_lib}")
            ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
        self.bb = _TrtRunner(backbone_engine, self.device)
        assert len(self.bb.input_names) == 1, f"backbone expects 1 input, got {self.bb.input_names}"
        self._bb_in = self.bb.input_names[0]
        self._bb_out = self.bb.output_names[0]

        self.head = None
        if head_engine and os.path.exists(head_engine):
            try:
                self.head = _TrtRunner(head_engine, self.device)
            except Exception as e:
                print(f"[TrtBackend] head engine load failed ({e}); falling back to head_fallback")
                self.head = None
        else:
            print(f"[TrtBackend] head engine not provided/missing → head_fallback active")

    def run_backbone(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 5:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
        else:
            B, N = 1, img.size(0)
        outs = self.bb.run({self._bb_in: img})
        out = outs[self._bb_out]
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
        if self.head is None:
            if self.head_fallback is None:
                raise RuntimeError(
                    "TrtBackend.run_head: head engine not loaded and no head_fallback."
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
        # Drop any not declared by the engine (defensive — should all be present).
        feeds = {k: v for k, v in feeds.items() if k in self.head.input_names}
        outs = self.head.run(feeds)
        # Reorder by engine declaration to OUT_NAMES.
        # Heuristic: engine output names follow 'out.bev_embed' etc.
        head_outs_ordered = []
        for n in OUT_NAMES:
            key = 'out.' + n
            if key in outs:
                head_outs_ordered.append(outs[key])
            elif n in outs:
                head_outs_ordered.append(outs[n])
            else:
                # Last resort: positional by engine declared order.
                head_outs_ordered.append(outs[self.head.output_names[len(head_outs_ordered)]])
        return dict(zip(OUT_NAMES, head_outs_ordered))

    def close(self) -> None:
        self.bb = None
        self.head = None
