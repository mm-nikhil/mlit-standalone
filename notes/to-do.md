# To-Do

1) Document baseline pipeline
- Identify the exact payload IR variants under `test/Standalone/` to start with.
- Confirm transform entry points and a minimal tile+fuse schedule.

2) Build extensible test suite
- Add more matmul+add+relu variants (operand order, relu forms, extra elementwise).
- Add FileCheck patterns for key structural changes (tiling loops, fusion).

3) Matching strategy (Transform dialect)
- Use matcher sequences to find `linalg.matmul` + consumer chains.
- Prototype `linalg.generic` matmul detection (rank + body math + indexing maps).
- Decide if custom matcher op or PDL-based matcher is needed.

4) Tiling strategy design
- Define a parameterized tiling policy (outer/L2, inner/L1).
- Encode two-level tiling + fusion steps in Transform IR.
- Add a config mechanism (target = x86 vs accelerator; tile sizes per target).

5) Benchmark harness
- Run transform on x86 locally, capture timing + output IR.
- Add knobs to sweep tile sizes and compare results.

6) Accelerator path
- Map tiling policy to L2/L1 constraints (fabric size, CR/CE layout).
- Extend Transform schedule to emit accelerator-friendly tiling structure.
