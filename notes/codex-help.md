# codex-help

Any learnings during execution should be added in Notable learnings section.

## Project intent (summary)
- Build a Transform-dialect-based tiling+fusion strategy for matmul+fc+relu graphs.
- Start with x86, then add hierarchical tiling for custom accelerator (L2 then L1).
- Match both `linalg.matmul` and matmul-like `linalg.generic`.

## Key files
- `mlir-standalone/standalone-opt/standalone-opt.cpp`
- `mlir-standalone/standalone-opt/CMakeLists.txt`
- `mlir-standalone/test/Standalone/* -- for test cases`
- `mlir-standalone/tools/llvm-lit-serial.py`
- `mlir-standalone/notes/requirements.md`
- `mlir-standalone/notes/to-do.md`

## Build/run commands
- Configure:
  `cmake -G Ninja -S /home/ws-mm-69/workspace/mlir-standalone -B /home/ws-mm-69/workspace/mlir-standalone/build \
   -DMLIR_DIR=/home/ws-mm-69/workspace/llvm-project/build/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=/home/ws-mm-69/workspace/mlir-standalone/tools/llvm-lit-serial.py`
- Build:
  `cmake --build /home/ws-mm-69/workspace/mlir-standalone/build`
- Run transform:
  `.../build/bin/standalone-opt -transform-interpreter test/Standalone/matmul-fc-relu-transformed.mlir`
- Run tests:
  `cmake --build /home/ws-mm-69/workspace/mlir-standalone/build --target check-standalone-opt`

## Notable learnings
- Transform interpreter consumes handles; don’t reuse consumed handles.
- `transform.collect_matching` requires matcher arguments to be `{transform.readonly}`.
- Use `transform.get_parent_op` on the *fused* handle if you need a post-fusion parent.
- This environment blocks Python multiprocessing semaphores; use the serial lit wrapper.
- Local x86 target is i5-13420H (8 cores/12 threads); caches: L1d 40KB/core, L2 1.3MB per cache instance (7MB total), L3 12MB shared; AVX2+FMA available.
- For f32 512x512 matmul, reasonable starting tiles are L2: M=N=128, K=64 or 128; L1: M=N=32 or 64, K=32 (multiples of 8 for AVX2).
- `transform-interpreter` supports `entry-point=` and `debug-bind-trailing-args=` for selecting policies and binding params.

## References (local)
- Transform tutorial: `llvm-project/mlir/docs/Tutorials/transform/_index.md`
- Matching guide: `llvm-project/mlir/docs/Tutorials/transform/Ch4.md`
- Transform dialect overview: `llvm-project/mlir/docs/Dialects/Transform.md`
