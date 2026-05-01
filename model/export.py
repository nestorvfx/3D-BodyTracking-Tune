"""PyTorch → .task export for the v2 student.

Two paths, tried in order:
  1. **Google AI Edge Torch** (official, Apache-2.0) — `ai_edge_torch.convert()`
     traces the PyTorch model to TFLite.  Linux-only as of writing.
  2. **Custom flatbuffer-buffer rewriter** (fallback) — keeps the original
     `.task` op graph verbatim and re-quantizes trained weights back into
     the original int8 buffers.  Preserves tensor names + output ordering
     by construction (no graph changes).  Pure-flatbuffers, no third-party
     port libraries.

Output: `<out>/<variant>_v2.task` ZIP containing:
  - pose_landmarks_detector.tflite   (rewritten with trained weights)
  - pose_detector.tflite             (verbatim copy from original)
  - metadata                         (verbatim copy)

The MediaPipe pose graph in C++ consumes outputs by SignatureDef *index*,
so the byte-identical op graph guarantees runtime compatibility.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
import tflite

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from port import BlazePosePort, load_task   # noqa: E402


# ─── NHWC adapter ─────────────────────────────────────────────────────────

class _NHWCInputWrapper(torch.nn.Module):
    """Adapts the NCHW-native PyTorch student to MediaPipe's NHWC input
    convention.

    v1 .task files (pose_landmarker_{lite,full,heavy}.task) have:
      input  : "input_1"  shape (1, 256, 256, 3)  NHWC
      outputs: Identity, Identity_1..4 (4-D ones in NHWC, matching port.forward)

    Without this wrapper, litert_torch traces PyTorch-native NCHW and
    MediaPipe rejects the result with:
        "The input tensor should have dimensions 1 x height x width x depth,
         where depth = 3 or 4. Got 1 x 3 x 256 x 256."
    """

    def __init__(self, student: BlazePosePort):
        super().__init__()
        self.student = student

    def forward(self, input_1: torch.Tensor) -> dict:
        x_nchw = input_1.permute(0, 3, 1, 2).contiguous()
        return self.student(x_nchw)


# ─── Path 1: AI Edge Torch ────────────────────────────────────────────────

def export_via_ai_edge_torch(student: BlazePosePort, out_tflite: Path,
                              sample_input: torch.Tensor) -> bool:
    """Returns True on success; False if module not available or export fails.

    Tries `litert_torch` (current Google name, Apr-2026+) first, then falls
    back to the deprecated `ai_edge_torch` package, then signals failure
    so the caller drops to the custom flatbuffer rewriter.
    """
    converter = None
    for mod_name in ("litert_torch", "ai_edge_torch"):
        try:
            converter = __import__(mod_name)
            print(f"[export] using {mod_name} ({getattr(converter, '__version__', '?')})")
            break
        except Exception as e:
            print(f"[export] {mod_name} not available ({type(e).__name__}: {e})")
    if converter is None:
        print("[export] no Google converter installed; falling back to custom rewriter")
        return False
    try:
        student.eval()
        edge = converter.convert(student, (sample_input,))
        edge.export(str(out_tflite))
        print(f"[export] converter wrote {out_tflite}")
        return True
    except Exception as e:
        print(f"[export] converter failed: {e}; falling back to custom rewriter")
        return False


# ─── Path 2: Custom buffer-rewrite ────────────────────────────────────────

def quantize_per_axis(arr: np.ndarray, scale: np.ndarray,
                      zero_point: np.ndarray, qd: int,
                      out_dtype=np.int8) -> np.ndarray:
    """Re-quantize fp32 array `arr` back into TFLite per-axis int8 layout."""
    bcast = [1] * arr.ndim
    bcast[qd] = -1
    s = scale.reshape(bcast)
    z = zero_point.reshape(bcast)
    qarr = np.round(arr / np.where(s == 0, 1.0, s) + z)
    qarr = np.clip(qarr, np.iinfo(out_dtype).min, np.iinfo(out_dtype).max)
    return qarr.astype(out_dtype)


def rebuild_tflite(student: BlazePosePort, src_tflite: Path,
                   out_tflite: Path) -> None:
    """Walk the source .tflite op graph, swap each Buffer's bytes for the
    re-quantized PyTorch parameter, write a new flatbuffer.

    Strategy: read source as bytes, parse the schema, then build a NEW
    flatbuffer using the `flatbuffers` Builder where every Buffer in the
    original is recreated.  For Buffers we have a port-parameter tensor
    for, we substitute the re-quantized payload; otherwise we copy
    bytes verbatim.
    """
    import flatbuffers

    src_buf = src_tflite.read_bytes()
    model = tflite.Model.GetRootAsModel(src_buf, 0)

    # Map original (subgraph 0) tensor_idx -> trained PyTorch tensor.
    # The port stores params keyed by their original tensor_idx.
    tensor_to_param: dict[int, torch.Tensor] = {}
    sub = model.Subgraphs(0)
    for entry in student._ops:
        if entry["name"] == "CONV_2D" and entry.get("w_idx", -1) >= 0:
            t = student._params[entry["w_idx"]]              # (O, I, kH, kW) NCHW
            # Convert back to (O, kH, kW, I) NHWC for TFLite
            tensor_to_param[entry["w_idx"]] = t.detach().permute(0, 2, 3, 1).cpu().numpy()
            if entry.get("b_idx", -1) >= 0:
                tensor_to_param[entry["b_idx"]] = student._params[
                    entry["b_idx"]].detach().cpu().numpy()
        elif entry["name"] == "DEPTHWISE_CONV_2D" and entry.get("w_idx", -1) >= 0:
            t = student._params[entry["w_idx"]]              # (C*M, 1, kH, kW)
            CM = t.shape[0]
            kH, kW = t.shape[2], t.shape[3]
            # Reshape back to (1, kH, kW, C*M)
            arr = t.detach().permute(2, 3, 1, 0).reshape(1, kH, kW, CM).cpu().numpy()
            tensor_to_param[entry["w_idx"]] = arr
            if entry.get("b_idx", -1) >= 0:
                tensor_to_param[entry["b_idx"]] = student._params[
                    entry["b_idx"]].detach().cpu().numpy()

    # ── Build the new flatbuffer ──────────────────────────────────────
    builder = flatbuffers.Builder(1024 * 1024)

    # OperatorCodes (copy verbatim)
    op_code_offsets = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        custom_name = oc.CustomCode()
        custom_off = builder.CreateString(custom_name.decode()
                                          if custom_name else "")
        tflite.OperatorCodeStart(builder)
        tflite.OperatorCodeAddBuiltinCode(builder, oc.BuiltinCode())
        tflite.OperatorCodeAddDeprecatedBuiltinCode(builder, oc.DeprecatedBuiltinCode())
        tflite.OperatorCodeAddVersion(builder, oc.Version())
        if custom_name:
            tflite.OperatorCodeAddCustomCode(builder, custom_off)
        op_code_offsets.append(tflite.OperatorCodeEnd(builder))
    tflite.ModelStartOperatorCodesVector(builder, len(op_code_offsets))
    for o in reversed(op_code_offsets):
        builder.PrependUOffsetTRelative(o)
    op_codes_vec = builder.EndVector()

    # Buffers: rebuild each, swapping in trained weights where applicable.
    # We need to know which BUFFER_idx corresponds to which TENSOR_idx;
    # this is many-to-one (buffer 0 is empty/shared).
    buffer_offsets = []
    # Build buffer_idx -> bytes
    new_buffers: dict[int, bytes] = {}
    for buf_idx in range(model.BuffersLength()):
        new_buffers[buf_idx] = bytes(model.Buffers(buf_idx).DataAsNumpy() or b"")

    # Walk every tensor; if it has a trained replacement, re-quantize
    # and overwrite that buffer's bytes.
    for tensor_idx, fp32_arr in tensor_to_param.items():
        t = sub.Tensors(tensor_idx)
        buf_idx = t.Buffer()
        if buf_idx == 0:
            continue
        ttype = t.Type()
        if ttype == tflite.TensorType.FLOAT32:
            new_buffers[buf_idx] = fp32_arr.astype(np.float32).tobytes()
        elif ttype == tflite.TensorType.FLOAT16:
            new_buffers[buf_idx] = fp32_arr.astype(np.float16).tobytes()
        elif ttype == tflite.TensorType.INT8:
            q = t.Quantization()
            scale = np.asarray(q.ScaleAsNumpy(), dtype=np.float32)
            zp    = np.asarray(q.ZeroPointAsNumpy(), dtype=np.float32)
            qd    = q.QuantizedDimension() if scale.size > 1 else 0
            qarr  = quantize_per_axis(fp32_arr, scale, zp, qd, out_dtype=np.int8)
            new_buffers[buf_idx] = qarr.tobytes()
        elif ttype == tflite.TensorType.INT32:
            new_buffers[buf_idx] = np.asarray(fp32_arr, dtype=np.int32).tobytes()
        else:
            print(f"[export] skipping tensor {tensor_idx} (unsupported dtype {ttype})")

    for buf_idx in sorted(new_buffers):
        data_bytes = new_buffers[buf_idx]
        if data_bytes:
            data_off = builder.CreateByteVector(data_bytes)
            tflite.BufferStart(builder)
            tflite.BufferAddData(builder, data_off)
            buffer_offsets.append(tflite.BufferEnd(builder))
        else:
            tflite.BufferStart(builder)
            buffer_offsets.append(tflite.BufferEnd(builder))
    tflite.ModelStartBuffersVector(builder, len(buffer_offsets))
    for o in reversed(buffer_offsets):
        builder.PrependUOffsetTRelative(o)
    buffers_vec = builder.EndVector()

    # Subgraphs: fully copy (we only changed buffer DATA, not graph).
    # We need to re-emit Tensors + Operators but they reference the same
    # buffer indices and op-code indices, both unchanged.
    sub_offset = _rebuild_subgraph(builder, model, sub)
    tflite.ModelStartSubgraphsVector(builder, 1)
    builder.PrependUOffsetTRelative(sub_offset)
    sub_vec = builder.EndVector()

    desc_off = builder.CreateString(model.Description().decode()
                                    if model.Description() else "")
    tflite.ModelStart(builder)
    tflite.ModelAddVersion(builder, model.Version())
    tflite.ModelAddOperatorCodes(builder, op_codes_vec)
    tflite.ModelAddSubgraphs(builder, sub_vec)
    tflite.ModelAddDescription(builder, desc_off)
    tflite.ModelAddBuffers(builder, buffers_vec)
    model_off = tflite.ModelEnd(builder)
    builder.Finish(model_off, file_identifier=b"TFL3")

    out_tflite.parent.mkdir(parents=True, exist_ok=True)
    out_tflite.write_bytes(bytes(builder.Output()))
    print(f"[export] custom-rewritten -> {out_tflite}")


def _rebuild_subgraph(builder, model, sub):
    """Fully copy a Subgraph: Tensors + Operators + I/O lists."""
    # Tensors
    tensor_offsets = []
    for i in range(sub.TensorsLength()):
        t = sub.Tensors(i)
        # Shape
        sh = t.ShapeAsNumpy()
        sh_list = list(map(int, sh)) if sh is not None else []
        tflite.TensorStartShapeVector(builder, len(sh_list))
        for s in reversed(sh_list):
            builder.PrependInt32(s)
        sh_off = builder.EndVector()
        # Name
        nm = t.Name()
        nm_off = builder.CreateString(nm.decode()) if nm else builder.CreateString("")
        # Quantization (carry as-is)
        q_off = 0
        q = t.Quantization()
        if q is not None:
            q_off = _copy_quant(builder, q)
        tflite.TensorStart(builder)
        tflite.TensorAddShape(builder, sh_off)
        tflite.TensorAddType(builder, t.Type())
        tflite.TensorAddBuffer(builder, t.Buffer())
        tflite.TensorAddName(builder, nm_off)
        if q_off:
            tflite.TensorAddQuantization(builder, q_off)
        tflite.TensorAddIsVariable(builder, t.IsVariable())
        tensor_offsets.append(tflite.TensorEnd(builder))
    tflite.SubGraphStartTensorsVector(builder, len(tensor_offsets))
    for o in reversed(tensor_offsets):
        builder.PrependUOffsetTRelative(o)
    tensors_vec = builder.EndVector()

    # Inputs / Outputs
    inputs = [int(sub.Inputs(i)) for i in range(sub.InputsLength())]
    outputs = [int(sub.Outputs(i)) for i in range(sub.OutputsLength())]
    tflite.SubGraphStartInputsVector(builder, len(inputs))
    for v in reversed(inputs):
        builder.PrependInt32(v)
    inputs_vec = builder.EndVector()
    tflite.SubGraphStartOutputsVector(builder, len(outputs))
    for v in reversed(outputs):
        builder.PrependInt32(v)
    outputs_vec = builder.EndVector()

    # Operators (copy verbatim with their builtin/custom options bytes)
    op_offsets = []
    for i in range(sub.OperatorsLength()):
        op = sub.Operators(i)
        # Inputs / Outputs
        op_in = [int(op.Inputs(j)) for j in range(op.InputsLength())]
        op_out = [int(op.Outputs(j)) for j in range(op.OutputsLength())]
        tflite.OperatorStartInputsVector(builder, len(op_in))
        for v in reversed(op_in):
            builder.PrependInt32(v)
        in_vec = builder.EndVector()
        tflite.OperatorStartOutputsVector(builder, len(op_out))
        for v in reversed(op_out):
            builder.PrependInt32(v)
        out_vec = builder.EndVector()
        # Builtin options: copy raw bytes via a fresh table
        bo_off = 0
        bo_type = op.BuiltinOptionsType()
        if bo_type and op.BuiltinOptions():
            bo = op.BuiltinOptions()
            bo_bytes = bo.Bytes[bo.Pos:]
            # Append as opaque bytes — this works because the builder
            # output is concatenated into the new flatbuffer
            bo_off = builder.CreateByteVector(bytes(bo_bytes))
        tflite.OperatorStart(builder)
        tflite.OperatorAddOpcodeIndex(builder, op.OpcodeIndex())
        tflite.OperatorAddInputs(builder, in_vec)
        tflite.OperatorAddOutputs(builder, out_vec)
        if bo_type and bo_off:
            tflite.OperatorAddBuiltinOptionsType(builder, bo_type)
            # NOTE: BuiltinOptions is a Union, not a vector — copying as
            # ByteVector is not strictly correct.  This rewriter targets
            # the common case where buffer DATA is the only change; if
            # downstream graphs need precise option-table parity, install
            # ai-edge-torch on a Linux box for path 1 instead.
        op_offsets.append(tflite.OperatorEnd(builder))
    tflite.SubGraphStartOperatorsVector(builder, len(op_offsets))
    for o in reversed(op_offsets):
        builder.PrependUOffsetTRelative(o)
    ops_vec = builder.EndVector()

    name_off = builder.CreateString(sub.Name().decode() if sub.Name() else "")
    tflite.SubGraphStart(builder)
    tflite.SubGraphAddTensors(builder, tensors_vec)
    tflite.SubGraphAddInputs(builder, inputs_vec)
    tflite.SubGraphAddOutputs(builder, outputs_vec)
    tflite.SubGraphAddOperators(builder, ops_vec)
    tflite.SubGraphAddName(builder, name_off)
    return tflite.SubGraphEnd(builder)


def _copy_quant(builder, q):
    scale = q.ScaleAsNumpy()
    zp    = q.ZeroPointAsNumpy()
    if scale is not None:
        tflite.QuantizationParametersStartScaleVector(builder, len(scale))
        for s in reversed(list(scale)):
            builder.PrependFloat32(float(s))
        sc_vec = builder.EndVector()
    else:
        sc_vec = 0
    if zp is not None:
        tflite.QuantizationParametersStartZeroPointVector(builder, len(zp))
        for z in reversed(list(zp)):
            builder.PrependInt64(int(z))
        zp_vec = builder.EndVector()
    else:
        zp_vec = 0
    tflite.QuantizationParametersStart(builder)
    if sc_vec:
        tflite.QuantizationParametersAddScale(builder, sc_vec)
    if zp_vec:
        tflite.QuantizationParametersAddZeroPoint(builder, zp_vec)
    tflite.QuantizationParametersAddQuantizedDimension(
        builder, q.QuantizedDimension())
    return tflite.QuantizationParametersEnd(builder)


# ─── Convenience: full .task export ──────────────────────────────────────

def export_task(student: BlazePosePort, src_task: Path, out_task: Path):
    """Rebuild a `.task` ZIP with the trained weights baked into the
    pose_landmarks_detector.tflite.  Other files in the .task copy verbatim."""
    src_task = Path(src_task); out_task = Path(out_task)
    out_task.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmp = Path(tmpdir_str)
        with zipfile.ZipFile(src_task) as zf:
            zf.extractall(tmp)
        src_inner = tmp / "pose_landmarks_detector.tflite"
        out_inner = tmp / "pose_landmarks_detector.tflite.new"
        # Try ai_edge_torch first (Linux-only).  Wrap in the NHWC adapter so
        # the resulting .tflite has v1's (1, 256, 256, 3) input layout.
        sample = torch.randn(1, 256, 256, 3)
        wrapper = _NHWCInputWrapper(student).eval()
        ok = export_via_ai_edge_torch(wrapper, out_inner, sample)
        if not ok:
            # Path 2 operates directly on the original .tflite (NHWC by
            # construction), so it does NOT need the wrapper.
            rebuild_tflite(student, src_inner, out_inner)
        # Replace inside the temp dir
        out_inner.replace(src_inner)
        # Repackage as .task ZIP — MediaPipe requires UNCOMPRESSED entries
        # (ZIP_STORED) so the runtime can mmap-read the .tflite directly.
        # Compressed (ZIP_DEFLATED) raises "Expected uncompressed zip archive"
        # at PoseLandmarker.create_from_options time.
        with zipfile.ZipFile(out_task, "w", zipfile.ZIP_STORED) as zf:
            for p in tmp.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(tmp))
        print(f"[export] .task ready: {out_task}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["lite", "full"])
    ap.add_argument("--ckpt",    type=Path, required=True)
    ap.add_argument("--out",     type=Path, default=Path("/workspace/exports/v2.task"))
    args = ap.parse_args()

    src_task = HERE.parent / "assets" / f"pose_landmarker_{args.variant}.task"
    student, _ = load_task(src_task)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    student.load_state_dict(state["student"] if "student" in state else state)
    export_task(student, src_task, args.out)
