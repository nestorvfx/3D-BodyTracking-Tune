"""Phase-1 inspection: unpack a .task and dump op graph + tensor shapes.

Usage:
    python inspect_task.py path/to/pose_landmarker_lite.task

Writes:
    inspection/<stem>_summary.txt   high-level overview
    inspection/<stem>_ops.txt       full op-by-op listing with shapes
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import tflite
import numpy as np

HERE = Path(__file__).resolve().parent


def opname(model: tflite.Model, op: tflite.Operator) -> str:
    code = model.OperatorCodes(op.OpcodeIndex())
    builtin = code.BuiltinCode()
    # tflite 2.18 maps builtin codes via tflite.BuiltinOperator
    for name in dir(tflite.BuiltinOperator):
        if name.startswith("_"):
            continue
        if getattr(tflite.BuiltinOperator, name) == builtin:
            return name
    return f"BUILTIN_{builtin}"


def fmt_shape(shape) -> str:
    if shape is None or len(shape) == 0:
        return "()"
    return "x".join(str(int(s)) for s in shape)


def inspect_tflite(tflite_path: Path, ops_out: Path, summary_out: Path):
    buf = tflite_path.read_bytes()
    model = tflite.Model.GetRootAsModel(buf, 0)
    n_subs = model.SubgraphsLength()
    if n_subs != 1:
        print(f"[warn] {tflite_path.name} has {n_subs} subgraphs; using #0")
    sub = model.Subgraphs(0)

    # Tensor table
    tensors = []
    for i in range(sub.TensorsLength()):
        t = sub.Tensors(i)
        sh = t.ShapeAsNumpy()
        tensors.append({
            "idx":  i,
            "name": t.Name().decode("utf-8") if t.Name() else f"tensor_{i}",
            "shape": list(sh) if sh is not None else [],
            "type": t.Type(),
            "buffer_idx": t.Buffer(),
            "is_constant": model.Buffers(t.Buffer()).DataLength() > 0,
        })

    # I/O tensor lists
    inputs  = [tensors[sub.Inputs(i)]  for i in range(sub.InputsLength())]
    outputs = [tensors[sub.Outputs(i)] for i in range(sub.OutputsLength())]

    # Op count + opcode histogram
    opcode_hist: dict[str, int] = {}
    op_lines: list[str] = []
    op_lines.append("# op_idx  op_name                          input_shapes -> output_shapes\n")
    for i in range(sub.OperatorsLength()):
        op = sub.Operators(i)
        name = opname(model, op)
        opcode_hist[name] = opcode_hist.get(name, 0) + 1
        ins  = [tensors[op.Inputs(j)]  for j in range(op.InputsLength())
                if op.Inputs(j) >= 0 and not tensors[op.Inputs(j)]["is_constant"]]
        outs = [tensors[op.Outputs(j)] for j in range(op.OutputsLength())]
        cins = [tensors[op.Inputs(j)]  for j in range(op.InputsLength())
                if op.Inputs(j) >= 0 and tensors[op.Inputs(j)]["is_constant"]]
        in_shapes  = [fmt_shape(t["shape"]) for t in ins]
        out_shapes = [fmt_shape(t["shape"]) for t in outs]
        const_shapes = [fmt_shape(t["shape"]) for t in cins]
        line = (f"{i:4d}  {name:30s}  in={in_shapes}  -> out={out_shapes}"
                f"  consts={const_shapes}")
        op_lines.append(line + "\n")

    ops_out.write_text("".join(op_lines))

    # Summary
    sum_lines = []
    sum_lines.append(f"=== {tflite_path.name} ===\n")
    sum_lines.append(f"size: {tflite_path.stat().st_size / 1024:.1f} KB\n")
    sum_lines.append(f"subgraphs: {n_subs}\n")
    sum_lines.append(f"operators: {sub.OperatorsLength()}\n")
    sum_lines.append(f"tensors:   {sub.TensorsLength()}\n")
    sum_lines.append(f"\nINPUT tensors:\n")
    for t in inputs:
        sum_lines.append(f"  [{t['idx']:4d}] {t['name']:40s} shape={fmt_shape(t['shape']):15s} type={t['type']}\n")
    sum_lines.append(f"\nOUTPUT tensors:\n")
    for t in outputs:
        sum_lines.append(f"  [{t['idx']:4d}] {t['name']:40s} shape={fmt_shape(t['shape']):15s} type={t['type']}\n")
    sum_lines.append(f"\nOpcode histogram:\n")
    for name, n in sorted(opcode_hist.items(), key=lambda kv: -kv[1]):
        sum_lines.append(f"  {name:30s} {n}\n")
    summary_out.write_text("".join(sum_lines))
    print("".join(sum_lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task_paths", type=Path, nargs="+")
    p.add_argument("--out", type=Path, default=HERE / "inspection")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for tp in args.task_paths:
        with zipfile.ZipFile(tp) as zf:
            stem = tp.stem
            unpacked = args.out / f"{stem}_unpacked"
            unpacked.mkdir(exist_ok=True)
            zf.extractall(unpacked)
            print(f"\n--- {tp.name} contents:")
            for n in zf.namelist():
                print(f"   {n}")
            for inner in unpacked.glob("*.tflite"):
                inspect_tflite(
                    inner,
                    args.out / f"{stem}__{inner.stem}_ops.txt",
                    args.out / f"{stem}__{inner.stem}_summary.txt",
                )


if __name__ == "__main__":
    main()
