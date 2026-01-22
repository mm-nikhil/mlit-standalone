# Requirements

- Benchmark Transform-dialect tiling+fusion on matmul+fc+relu style graphs.
- Support extensible test shapes/graphs (e.g., relu(add(matmul(x,y), bias), 0), relu(0, add(...)), swapped operand orders).
- Transform strategy must recognize:
  - `linalg.matmul` directly.
  - `linalg.generic` that semantically implements matmul (validate ranks, indexing maps, body math).
- Provide a configurable, target-specific tiling policy:
  - x86 baseline (local runs).
  - custom accelerator with hierarchical tiling (L2 then L1, CR/CE layout).
- Multi-step tiling pipeline (outer/L2 tiles, inner/L1 tiles) + producer fusion.
- Clear benchmarking harness and metrics capture (timing, tile sizes, IR diffs).
