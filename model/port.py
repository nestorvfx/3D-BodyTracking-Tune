"""Custom TFLite -> PyTorch port for BlazePose pose-landmark networks.

Walks the `.tflite` op graph from a `.task` ZIP, instantiates a generic
PyTorch nn.Module that re-runs the same ops in PyTorch, and copies the
quantized constants in dequantized form into the model.

Apache-2.0 licensed inputs only:
- `pose_landmarker_{lite,full,heavy}.task` — Apache-2.0 (Google Model Card)
- `tflite` PyPI package — schema parser, Apache-2.0
- PyTorch — BSD-3

No third-party port libraries used (zmurez/MediaPipePyTorch, nobuco,
tflite2pytorch, mediapipe-pytorch — all explicitly forbidden by user).

Internal representation: NCHW (PyTorch native).  Permute at the input
(NHWC -> NCHW) and at any 4-D output (NCHW -> NHWC) so the byte-equivalence
check sees TFLite-format tensors.

Supported ops (the union across Lite/Full/Heavy):
  PAD, DEQUANTIZE, CONV_2D, DEPTHWISE_CONV_2D, ADD, RESHAPE,
  RESIZE_BILINEAR, MAX_POOL_2D, LOGISTIC.

Failure mode: any unrecognised op raises NotImplementedError loudly with
the op name + index — user adds the handler.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import tflite
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Enums ────────────────────────────────────────────────────────────────
ACT_NONE = 0
ACT_RELU = 1
ACT_RELU_N1_TO_1 = 2
ACT_RELU6 = 3
ACT_TANH = 4

PAD_SAME = 0
PAD_VALID = 1


def _opname(model: tflite.Model, op: tflite.Operator) -> str:
    builtin = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
    for n in dir(tflite.BuiltinOperator):
        if n.startswith("_"):
            continue
        if getattr(tflite.BuiltinOperator, n) == builtin:
            return n
    return f"BUILTIN_{builtin}"


def _builtin_options(op: tflite.Operator, options_class):
    """Decode op.BuiltinOptions() as the given Options class."""
    table = op.BuiltinOptions()
    if table is None:
        return None
    obj = options_class()
    obj.Init(table.Bytes, table.Pos)
    return obj


def _apply_activation(x: torch.Tensor, act: int) -> torch.Tensor:
    if act == ACT_NONE:    return x
    if act == ACT_RELU:    return F.relu(x)
    if act == ACT_RELU6:   return F.relu6(x)
    if act == ACT_RELU_N1_TO_1: return torch.clamp(x, -1.0, 1.0)
    if act == ACT_TANH:    return torch.tanh(x)
    raise NotImplementedError(f"fused activation {act}")


# ─── Constant dequantization ──────────────────────────────────────────────

def dequantize_constant(tensor, model: tflite.Model) -> np.ndarray:
    """Return fp32 ndarray for a constant tensor, applying TFLite quant if needed."""
    buf = model.Buffers(tensor.Buffer())
    raw = buf.DataAsNumpy()
    if raw is None or raw.size == 0:
        return None
    shape = tensor.ShapeAsNumpy()
    shape = list(shape) if shape is not None else []
    ttype = tensor.Type()

    if ttype == tflite.TensorType.FLOAT32:
        return raw.view(np.float32).reshape(shape).astype(np.float32, copy=True)
    if ttype == tflite.TensorType.FLOAT16:
        return raw.view(np.float16).reshape(shape).astype(np.float32)
    if ttype == tflite.TensorType.INT32:
        return raw.view(np.int32).reshape(shape).astype(np.float32)

    if ttype in (tflite.TensorType.INT8, tflite.TensorType.UINT8):
        if ttype == tflite.TensorType.INT8:
            arr = raw.view(np.int8).reshape(shape)
        else:
            arr = raw.view(np.uint8).reshape(shape)
        q = tensor.Quantization()
        if q is None:
            return arr.astype(np.float32)
        scale = q.ScaleAsNumpy()
        zp    = q.ZeroPointAsNumpy()
        if scale is None or (hasattr(scale, "size") and scale.size == 0):
            return arr.astype(np.float32)
        scale = np.asarray(scale, dtype=np.float32)
        zp    = np.asarray(zp,    dtype=np.float32)
        if scale.size == 1:
            return (arr.astype(np.float32) - float(zp[0])) * float(scale[0])
        # Per-axis quant
        qd = q.QuantizedDimension()
        bcast = [1] * arr.ndim
        bcast[qd] = -1
        s = scale.reshape(bcast)
        z = zp.reshape(bcast)
        return (arr.astype(np.float32) - z) * s

    raise NotImplementedError(f"unsupported tensor type {ttype} for dequant")


# ─── The port ─────────────────────────────────────────────────────────────

class BlazePosePort(nn.Module):
    """Run a BlazePose pose-landmark .tflite as a PyTorch module.

    Parameters are exposed via nn.Parameter so they can be fine-tuned later;
    DEQUANTIZE ops are folded into parameter loading at __init__ time.
    """

    def __init__(self, tflite_path: Path):
        super().__init__()
        buf = Path(tflite_path).read_bytes()
        self.model = tflite.Model.GetRootAsModel(buf, 0)
        if self.model.SubgraphsLength() != 1:
            raise NotImplementedError("expected exactly 1 subgraph")
        self.sub = self.model.Subgraphs(0)
        self.input_idx  = [self.sub.Inputs(i)  for i in range(self.sub.InputsLength())]
        self.output_idx = [self.sub.Outputs(i) for i in range(self.sub.OutputsLength())]
        self._params: dict[int, nn.Parameter] = {}     # tensor_idx -> param (NCHW)
        self._consts: dict[int, torch.Tensor] = {}     # non-trainable scalars/shapes
        self._ops: list[dict[str, Any]] = []
        self._build()

    # ----- build -------------------------------------------------------
    def _build(self):
        sub = self.sub
        model = self.model
        # First pass: dequantize all constants and convert weight layouts.
        # We don't store raw tensors keyed by op output here — we handle
        # constants lazily per-op consumer (CONV_2D weights need OHWI->OIHW
        # perm; PAD's paddings need int32 directly).
        for i in range(sub.OperatorsLength()):
            op = sub.Operators(i)
            name = _opname(model, op)
            ins  = [op.Inputs(j)  for j in range(op.InputsLength())]
            outs = [op.Outputs(j) for j in range(op.OutputsLength())]
            entry: dict[str, Any] = {"i": i, "name": name, "ins": ins, "outs": outs}

            if name == "DEQUANTIZE":
                # consume: ins[0] is the int8/fp16 constant tensor
                t = sub.Tensors(ins[0])
                arr = dequantize_constant(t, model)
                if arr is None:
                    raise RuntimeError(f"DEQUANTIZE op {i}: empty constant")
                entry["const"] = arr
            elif name == "CONV_2D":
                opts = _builtin_options(op, tflite.Conv2DOptions)
                t_w = sub.Tensors(ins[1])
                t_b = sub.Tensors(ins[2]) if len(ins) > 2 else None
                # weights stored OHWI; convert to OIHW
                w = self._resolve_constant(ins[1])
                w_oihw = torch.from_numpy(w.transpose(0, 3, 1, 2).copy()).float()
                b = (torch.from_numpy(self._resolve_constant(ins[2])).float()
                     if t_b is not None else None)
                pname = f"conv_{i}_w"
                bname = f"conv_{i}_b"
                self._params[ins[1]] = nn.Parameter(w_oihw, requires_grad=True)
                self.register_parameter(pname, self._params[ins[1]])
                if b is not None:
                    self._params[ins[2]] = nn.Parameter(b, requires_grad=True)
                    self.register_parameter(bname, self._params[ins[2]])
                entry.update({
                    "stride": (opts.StrideH(), opts.StrideW()),
                    "padding": opts.Padding(),
                    "act": opts.FusedActivationFunction(),
                    "w_idx": ins[1], "b_idx": ins[2] if t_b is not None else -1,
                })
            elif name == "DEPTHWISE_CONV_2D":
                opts = _builtin_options(op, tflite.DepthwiseConv2DOptions)
                t_w = sub.Tensors(ins[1])
                t_b = sub.Tensors(ins[2]) if len(ins) > 2 else None
                w = self._resolve_constant(ins[1])    # shape (1, kH, kW, C*M)
                # Convert to (C, M, kH, kW): we use groups=C, depth_multiplier M
                kH, kW, CM = w.shape[1], w.shape[2], w.shape[3]
                M = opts.DepthMultiplier()
                C = CM // M
                w_arr = w.reshape(1, kH, kW, C, M).transpose(3, 4, 1, 2, 0)  # (C, M, kH, kW, 1)
                w_arr = w_arr.reshape(C * M, 1, kH, kW)
                w_torch = torch.from_numpy(w_arr.copy()).float()
                b = (torch.from_numpy(self._resolve_constant(ins[2])).float()
                     if t_b is not None else None)
                self._params[ins[1]] = nn.Parameter(w_torch, requires_grad=True)
                self.register_parameter(f"dwconv_{i}_w", self._params[ins[1]])
                if b is not None:
                    self._params[ins[2]] = nn.Parameter(b, requires_grad=True)
                    self.register_parameter(f"dwconv_{i}_b", self._params[ins[2]])
                entry.update({
                    "stride": (opts.StrideH(), opts.StrideW()),
                    "padding": opts.Padding(),
                    "act": opts.FusedActivationFunction(),
                    "groups": C,
                    "depth_multiplier": M,
                    "w_idx": ins[1], "b_idx": ins[2] if t_b is not None else -1,
                })
            elif name == "PAD":
                # ins[1] is the paddings constant (shape (rank, 2) int32)
                pads = self._resolve_constant(ins[1]).astype(np.int64)
                entry["pads"] = pads      # NHWC ordering
            elif name == "RESHAPE":
                # ins[1] is the new_shape constant (or use BuiltinOptions)
                new_shape = None
                opts = _builtin_options(op, tflite.ReshapeOptions)
                if opts is not None and opts.NewShapeLength() > 0:
                    new_shape = [opts.NewShape(j)
                                 for j in range(opts.NewShapeLength())]
                if new_shape is None and len(ins) > 1:
                    arr = self._resolve_constant(ins[1])
                    new_shape = arr.astype(int).tolist()
                entry["new_shape"] = new_shape
            elif name == "RESIZE_BILINEAR":
                opts = _builtin_options(op, tflite.ResizeBilinearOptions)
                size = self._resolve_constant(ins[1]).astype(int).tolist()
                entry["size"]              = tuple(size)   # (H, W)
                entry["align_corners"]     = bool(opts.AlignCorners())
                entry["half_pixel_centers"] = bool(opts.HalfPixelCenters())
            elif name == "MAX_POOL_2D":
                opts = _builtin_options(op, tflite.Pool2DOptions)
                entry.update({
                    "stride": (opts.StrideH(), opts.StrideW()),
                    "filter": (opts.FilterHeight(), opts.FilterWidth()),
                    "padding": opts.Padding(),
                    "act": opts.FusedActivationFunction(),
                })
            elif name == "ADD":
                opts = _builtin_options(op, tflite.AddOptions)
                entry["act"] = opts.FusedActivationFunction() if opts else 0
            elif name == "LOGISTIC":
                pass  # no options
            elif name == "CONCATENATION":
                opts = _builtin_options(op, tflite.ConcatenationOptions)
                entry["axis"] = opts.Axis()
                entry["act"]  = opts.FusedActivationFunction()
            elif name == "MEAN":
                # Reducer: ins[1] is axes tensor
                arr = self._resolve_constant(ins[1]).astype(int).tolist()
                opts = _builtin_options(op, tflite.ReducerOptions)
                entry["axes"] = arr
                entry["keep_dims"] = bool(opts.KeepDims()) if opts else False
            else:
                raise NotImplementedError(f"op {i}: unsupported '{name}'")
            self._ops.append(entry)

    def _resolve_constant(self, tensor_idx: int) -> np.ndarray:
        """Look up a tensor by index; if it's the OUTPUT of an earlier
        DEQUANTIZE we already prepared, return that.  Otherwise dequantize
        directly from its buffer."""
        for entry in self._ops:
            if entry.get("name") == "DEQUANTIZE" and tensor_idx in entry["outs"]:
                return entry["const"]
        # Direct constant
        t = self.sub.Tensors(tensor_idx)
        arr = dequantize_constant(t, self.model)
        if arr is None:
            raise RuntimeError(f"tensor {tensor_idx} has no constant data")
        return arr

    # ----- forward ----------------------------------------------------
    def forward(self, x: torch.Tensor):
        """x: (1, 3, 256, 256) NCHW float32 in [0, 1].

        Returns dict {tensor_name: torch.Tensor} matching the .tflite
        output names ('Identity', 'Identity_1', ...).  4-D outputs are
        permuted back to NHWC to match the .tflite reference.
        """
        sub = self.sub
        vals: dict[int, torch.Tensor] = {}
        # The .tflite expects NHWC input; we accept NCHW + auto-permute.
        if x.dim() == 4 and x.shape[1] == 3:
            vals[self.input_idx[0]] = x          # store as NCHW
        else:
            raise ValueError(f"expected (B,3,H,W) input, got {tuple(x.shape)}")

        for entry in self._ops:
            name = entry["name"]
            if name == "DEQUANTIZE":
                continue
            try:
                self._step(entry, vals)
            except Exception as e:
                in_shapes = [tuple(vals[i].shape) if i in vals else "MISSING"
                             for i in entry["ins"]]
                def _sh(idx):
                    a = self.sub.Tensors(idx).ShapeAsNumpy()
                    return list(map(int, a)) if a is not None else []
                schema_in  = [_sh(i) for i in entry["ins"]]
                schema_out = [_sh(o) for o in entry["outs"]]
                names = [self.sub.Tensors(i).Name() for i in entry["ins"]]
                raise RuntimeError(
                    f"op {entry['i']} {name}: {type(e).__name__}: {e}\n"
                    f"  runtime input shapes: {in_shapes}\n"
                    f"  schema input shapes:  {schema_in}\n"
                    f"  schema output shapes: {schema_out}\n"
                    f"  input names: {names}"
                ) from e

        # Map output tensor indices to their .tflite names + permute 4-D back.
        out: dict[str, torch.Tensor] = {}
        for tidx in self.output_idx:
            t = vals[tidx]
            t_obj = self.sub.Tensors(tidx)
            tname = (t_obj.Name().decode("utf-8")
                     if t_obj.Name() else f"out_{tidx}")
            if t.dim() == 4:
                t = t.permute(0, 2, 3, 1).contiguous()
            out[tname] = t
        return out

    def _step(self, entry, vals):
        """Apply a single TFLite op to the runtime tensor table `vals`."""
        name = entry["name"]
        if name == "PAD":
            t = vals[entry["ins"][0]]
            pads = entry["pads"]   # NHWC ordering: rows = (N, H, W, C)
            pad_h_pre, pad_h_post = int(pads[1, 0]), int(pads[1, 1])
            pad_w_pre, pad_w_post = int(pads[2, 0]), int(pads[2, 1])
            pad_c_pre = int(pads[3, 0]) if pads.shape[0] >= 4 else 0
            pad_c_post = int(pads[3, 1]) if pads.shape[0] >= 4 else 0
            # F.pad on (N,C,H,W): order is (W_l, W_r, H_t, H_b, C_pre, C_post, N_pre, N_post)
            vals[entry["outs"][0]] = F.pad(
                t, (pad_w_pre, pad_w_post,
                    pad_h_pre, pad_h_post,
                    pad_c_pre, pad_c_post))
        elif name == "CONV_2D":
            inp = vals[entry["ins"][0]]
            w = self._params[entry["w_idx"]]
            b = self._params.get(entry["b_idx"]) if entry["b_idx"] >= 0 else None
            sh, sw = entry["stride"]
            pad = 0 if entry["padding"] == PAD_VALID else self._same_pad(
                inp.shape[-2:], w.shape[-2:], (sh, sw), inp_pad=True)
            if isinstance(pad, tuple) and len(pad) == 4:
                inp = F.pad(inp, pad); pad = 0
            y = F.conv2d(inp, w, b, stride=(sh, sw), padding=pad)
            y = _apply_activation(y, entry["act"])
            vals[entry["outs"][0]] = y
        elif name == "DEPTHWISE_CONV_2D":
            inp = vals[entry["ins"][0]]
            w = self._params[entry["w_idx"]]
            b = self._params.get(entry["b_idx"]) if entry["b_idx"] >= 0 else None
            sh, sw = entry["stride"]
            pad = 0 if entry["padding"] == PAD_VALID else self._same_pad(
                inp.shape[-2:], w.shape[-2:], (sh, sw), inp_pad=True)
            if isinstance(pad, tuple) and len(pad) == 4:
                inp = F.pad(inp, pad); pad = 0
            y = F.conv2d(inp, w, b, stride=(sh, sw), padding=pad,
                         groups=entry["groups"])
            y = _apply_activation(y, entry["act"])
            vals[entry["outs"][0]] = y
        elif name == "ADD":
            a = vals[entry["ins"][0]]
            b = vals[entry["ins"][1]]
            y = a + b
            y = _apply_activation(y, entry.get("act", 0))
            vals[entry["outs"][0]] = y
        elif name == "MAX_POOL_2D":
            inp = vals[entry["ins"][0]]
            kh, kw = entry["filter"]
            sh, sw = entry["stride"]
            pad = 0 if entry["padding"] == PAD_VALID else self._same_pad(
                inp.shape[-2:], (kh, kw), (sh, sw), inp_pad=True)
            if isinstance(pad, tuple) and len(pad) == 4:
                inp = F.pad(inp, pad); pad = 0
            y = F.max_pool2d(inp, (kh, kw), (sh, sw), padding=pad)
            vals[entry["outs"][0]] = _apply_activation(y, entry["act"])
        elif name == "RESHAPE":
            inp = vals[entry["ins"][0]]
            new_shape = list(entry["new_shape"])
            if inp.dim() == 4:
                inp = inp.permute(0, 2, 3, 1).contiguous()
            vals[entry["outs"][0]] = inp.reshape(new_shape)
        elif name == "RESIZE_BILINEAR":
            inp = vals[entry["ins"][0]]
            size = entry["size"]
            y = F.interpolate(inp, size=size, mode="bilinear",
                              align_corners=bool(entry["align_corners"]))
            vals[entry["outs"][0]] = y
        elif name == "LOGISTIC":
            vals[entry["outs"][0]] = torch.sigmoid(vals[entry["ins"][0]])
        elif name == "CONCATENATION":
            inps = [vals[i] for i in entry["ins"]]
            axis = entry["axis"]
            if any(t.dim() == 4 for t in inps):
                nhwc = [t.permute(0, 2, 3, 1) for t in inps]
                y = torch.cat(nhwc, dim=axis).permute(0, 3, 1, 2).contiguous()
            else:
                y = torch.cat(inps, dim=axis)
            vals[entry["outs"][0]] = _apply_activation(y, entry["act"])
        elif name == "MEAN":
            inp = vals[entry["ins"][0]]
            axes = list(entry["axes"])
            if inp.dim() == 4:
                nhwc_to_nchw = {0: 0, 1: 2, 2: 3, 3: 1}
                axes = [nhwc_to_nchw[a] for a in axes]
            vals[entry["outs"][0]] = inp.mean(dim=axes,
                                              keepdim=entry["keep_dims"])
        else:
            raise NotImplementedError(f"runtime: op {name}")

    @staticmethod
    def _same_pad(input_hw, kernel_hw, stride_hw, inp_pad=False):
        """Compute SAME padding for TFLite-style asymmetric pad.
        Returns tuple (pl, pr, pt, pb) for F.pad if asymmetric, else int."""
        ih, iw = int(input_hw[0]), int(input_hw[1])
        kh, kw = int(kernel_hw[0]), int(kernel_hw[1])
        sh, sw = int(stride_hw[0]), int(stride_hw[1])
        # TFLite SAME: out = ceil(in / stride); pad_total = max(0, (out-1)*stride + k - in)
        oh = (ih + sh - 1) // sh
        ow = (iw + sw - 1) // sw
        pad_h = max(0, (oh - 1) * sh + kh - ih)
        pad_w = max(0, (ow - 1) * sw + kw - iw)
        pt = pad_h // 2
        pb = pad_h - pt
        pl = pad_w // 2
        pr = pad_w - pl
        if pt == pb and pl == pr:
            return (pt, pl)            # symmetric, use as F.conv2d padding=(pH, pW)
        # asymmetric — apply via F.pad
        return (pl, pr, pt, pb)


# ─── Convenience: load directly from a .task ──────────────────────────────

def load_task(task_path: Path) -> tuple[BlazePosePort, Path]:
    """Unzip the .task to a temp dir; return the BlazePosePort and the
    landmark .tflite path so byte-equivalence tests can use it."""
    task_path = Path(task_path)
    out = task_path.parent / f"{task_path.stem}_unpacked"
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(task_path) as z:
        z.extractall(out)
    landmark_tflite = out / "pose_landmarks_detector.tflite"
    if not landmark_tflite.exists():
        raise RuntimeError(f"no pose_landmarks_detector.tflite in {task_path}")
    return BlazePosePort(landmark_tflite), landmark_tflite


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("task", type=Path)
    args = p.parse_args()
    model, _ = load_task(args.task)
    print(f"Loaded {args.task.name}: "
          f"{sum(p.numel() for p in model.parameters())} params, "
          f"{len(model._ops)} ops")
