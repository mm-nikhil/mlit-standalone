#!/usr/bin/env python3
import argparse
import pathlib
import re
import subprocess
import sys
import time

TIMEOUT_S = 60


def run_cmd(cmd, *, stdout_path=None):
    try:
        if stdout_path is None:
            subprocess.run(cmd, check=True, timeout=TIMEOUT_S)
            return
        with open(stdout_path, "w", encoding="utf-8") as handle:
            subprocess.run(cmd, check=True, stdout=handle, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise SystemExit(f"timeout: command exceeded {TIMEOUT_S}s")


def build_pipeline(transform_opts=None, erase_transform=False):
    passes = []
    if transform_opts:
        opts = " ".join(transform_opts)
        passes += [
            f"transform-interpreter{{{opts}}}",
            "test-transform-dialect-erase-schedule",
        ]
    elif erase_transform:
        passes += ["test-transform-dialect-erase-schedule"]
    passes += [
        "one-shot-bufferize{bufferize-function-boundaries}",
        "buffer-deallocation-pipeline",
        "convert-bufferization-to-memref",
    ]
    passes += [
        "convert-linalg-to-loops",
        "scf-forall-to-for",
        "convert-scf-to-cf",
        "expand-strided-metadata",
        "lower-affine",
        "convert-ub-to-llvm",
        "convert-arith-to-llvm",
        "finalize-memref-to-llvm",
        "convert-func-to-llvm",
        "convert-cf-to-llvm",
        "reconcile-unrealized-casts",
    ]
    return f"--pass-pipeline=builtin.module({','.join(passes)})"


def lower_mlir(mlir_opt, input_path, output_path, pipeline_arg):
    cmd = [
        str(mlir_opt),
        str(input_path),
        pipeline_arg,
    ]
    start = time.perf_counter()
    run_cmd(cmd, stdout_path=output_path)
    return time.perf_counter() - start


def time_runner(mlir_runner, lowered_path, shared_libs, runs):
    cmd = [
        str(mlir_runner),
        str(lowered_path),
        "-e",
        "main",
        "-entry-point-result=void",
        f"-shared-libs={shared_libs}",
    ]
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise SystemExit(f"timeout: command exceeded {TIMEOUT_S}s")
        times.append(time.perf_counter() - start)
    return times


def parse_tilesize(value):
    parts = value.split(",")
    if len(parts) not in (3, 6):
        raise SystemExit(f"Invalid --tilesize value: {value}")
    return tuple(int(part.strip()) for part in parts)


def collect_entry_points(text):
    return set(re.findall(r"transform\.named_sequence\s+@([A-Za-z0-9_]+)", text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mlir_file", help="Path to the MLIR file to run.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--transform",
        action="store_true",
        help="Apply the Transform dialect schedule before lowering.",
    )
    parser.add_argument(
        "--entry-point",
        help="Transform entry point to run when --transform is set.",
    )
    parser.add_argument(
        "--tilesize",
        action="append",
        default=[],
        help="Comma-separated sizes (M,N,K) or (M2,N2,K2,M1,N1,K1) for multilevel.",
    )
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    llvm_bin = workspace_root / "llvm-project" / "build" / "bin"
    llvm_lib = workspace_root / "llvm-project" / "build" / "lib"

    mlir_opt = llvm_bin / "mlir-opt"
    mlir_runner = llvm_bin / "mlir-runner"
    shared_libs = f"{llvm_lib / 'libmlir_c_runner_utils.so'},{llvm_lib / 'libmlir_runner_utils.so'}"

    input_path = pathlib.Path(args.mlir_file)
    if not input_path.is_absolute():
        if input_path.exists():
            input_path = input_path.resolve()
        else:
            candidate = repo_root / input_path
            if candidate.exists():
                input_path = candidate.resolve()
    if not input_path.exists():
        raise SystemExit(f"MLIR file not found: {args.mlir_file}")

    tile_sizes = [parse_tilesize(value) for value in args.tilesize]
    if tile_sizes:
        tile_arity = len(tile_sizes[0])
        if any(len(tile) != tile_arity for tile in tile_sizes):
            raise SystemExit("All --tilesize values must use the same arity.")
    else:
        tile_arity = None
    if tile_sizes and not args.transform:
        args.transform = True

    input_text = input_path.read_text(encoding="utf-8")
    entry_points = collect_entry_points(input_text)
    has_transform_ir = bool(entry_points) or "transform.with_named_sequence" in input_text

    transform_entry = None
    erase_transform = has_transform_ir
    if args.transform:
        if not has_transform_ir:
            erase_transform = False
            transform_entry = None
            if tile_sizes:
                print(
                    "warning: --tilesize ignored because no transform IR was found.",
                    file=sys.stderr,
                )
                tile_sizes = []
            print(
                "warning: --transform set but no transform IR found; running without it.",
                file=sys.stderr,
            )
        else:
            if args.entry_point:
                transform_entry = args.entry_point
            elif tile_sizes:
                transform_entry = (
                    "tile_policy_multilevel" if tile_arity == 6 else "tile_policy"
                )
            else:
                transform_entry = "__transform_main"
            if transform_entry not in entry_points:
                if tile_sizes and "__transform_main" in entry_points:
                    print(
                        "warning: tile_policy not found; using __transform_main and "
                        "ignoring --tilesize.",
                        file=sys.stderr,
                    )
                    transform_entry = "__transform_main"
                    tile_sizes = []
                else:
                    available = ", ".join(sorted(entry_points)) or "<none>"
                    raise SystemExit(
                        f"Transform entry point '{transform_entry}' not found. "
                        f"Available: {available}"
                    )
            if transform_entry == "tile_policy" and not tile_sizes:
                raise SystemExit("entry point 'tile_policy' requires --tilesize M,N,K")
            if transform_entry == "tile_policy_multilevel" and not tile_sizes:
                raise SystemExit(
                    "entry point 'tile_policy_multilevel' requires --tilesize "
                    "M2,N2,K2,M1,N1,K1"
                )
            if tile_sizes:
                if transform_entry == "tile_policy" and tile_arity != 3:
                    raise SystemExit(
                        "entry point 'tile_policy' requires --tilesize M,N,K"
                    )
                if transform_entry == "tile_policy_multilevel" and tile_arity != 6:
                    raise SystemExit(
                        "entry point 'tile_policy_multilevel' requires --tilesize "
                        "M2,N2,K2,M1,N1,K1"
                    )

    out_dir = repo_root / "build"
    out_dir.mkdir(parents=True, exist_ok=True)

    def summarize(name, values):
        avg = sum(values) / len(values)
        best = min(values)
        print(f"{name}: avg={avg:.6f}s best={best:.6f}s runs={values}")

    if tile_sizes:
        for tile in tile_sizes:
            if tile_arity == 3:
                m_tile, n_tile, k_tile = tile
                name = f"m{m_tile}_n{n_tile}_k{k_tile}"
                args_str = f"{m_tile},#{n_tile},#{k_tile}"
                label = f"m={m_tile} n={n_tile} k={k_tile}"
            else:
                m2, n2, k2, m1, n1, k1 = tile
                name = f"m2{m2}_n2{n2}_k2{k2}_m1{m1}_n1{n1}_k1{k1}"
                args_str = f"{m2},#{n2},#{k2},#{m1},#{n1},#{k1}"
                label = f"m2={m2} n2={n2} k2={k2} m1={m1} n1={n1} k1={k1}"
            lowered_path = out_dir / f"{input_path.stem}_lowered_{name}.mlir"
            transform_opts = [
                f"entry-point={transform_entry}",
                f"debug-bind-trailing-args=#{args_str}",
            ]
            pipeline_arg = build_pipeline(
                transform_opts=transform_opts, erase_transform=erase_transform
            )
            compile_time = lower_mlir(mlir_opt, input_path, lowered_path, pipeline_arg)
            exec_times = time_runner(mlir_runner, lowered_path, shared_libs, args.runs)
            print(f"compile ({label}): {compile_time:.6f}s")
            summarize(f"exec ({label})", exec_times)
    else:
        lowered_path = out_dir / f"{input_path.stem}_lowered.mlir"
        transform_opts = (
            [f"entry-point={transform_entry}"] if transform_entry else None
        )
        pipeline_arg = build_pipeline(
            transform_opts=transform_opts, erase_transform=erase_transform
        )
        compile_time = lower_mlir(mlir_opt, input_path, lowered_path, pipeline_arg)
        exec_times = time_runner(mlir_runner, lowered_path, shared_libs, args.runs)
        print(f"compile: {compile_time:.6f}s")
        summarize("exec", exec_times)


if __name__ == "__main__":
    main()
