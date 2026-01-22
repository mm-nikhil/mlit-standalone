# Transform Dialect Notes + To-Do (mlir-standalone)

## What the Transform dialect is (mental model)
- The Transform dialect is a **control IR** that *describes* transformations to apply to a separate **payload IR** (your Linalg program). It is **not** a pass by itself.
- A **pass** (the transform interpreter) reads the Transform IR and applies its operations to the payload IR, rewriting it (e.g., tiling, fusion, vectorization).
- Transform ops operate on **handles** (typed references to payload ops/values/params). Handles are produced by match operations and consumed by transform ops.
- Transform ops have **side effects on a mapping resource** (they can “consume” handles). This is why there are passes like `transform-dialect-check-uses` and `transform-infer-effects`.
- Transform dialect can live in the same MLIR module as the payload or be **preloaded** as a library and applied to the payload.

## How it is used (IR + pass interaction)
- The Transform dialect itself is just IR. It doesn’t do anything until the **interpreter pass** runs.
- The interpreter pass is registered as `transform-interpreter` (see `mlir/Dialect/Transform/Transforms/Passes.td`).
- The interpreter looks for a **named sequence** entry point (default `__transform_main`) and applies it to the payload IR rooted at the pass anchor op (or a tagged root via `-debug-payload-root-tag`).
- Optional helper passes:
  - `transform-dialect-check-uses` to catch use-after-free of handles.
  - `transform-infer-effects` to infer side effects on handles.
  - `transform-preload-library` to load transform libraries from files.

## Linalg + Transform dialect specifics
- Linalg transform ops live in the **Linalg Transform dialect extension** (e.g., `transform.structured.tile_using_for`, `transform.structured.fuse`, etc.).
- Those ops are *not* part of core Transform dialect; you must **register the Linalg Transform extension** with the dialect registry.
- The extension is registered via `mlir::linalg::registerTransformDialectExtension(registry)` in `mlir/Dialect/Linalg/TransformOps/DialectExtension.h`.

## To-Do: Enable Transform dialect in mlir-standalone

### 1) Build prerequisites (LLVM/MLIR)
- Ensure your LLVM build includes the transform dialect and Linalg transform ops (they are part of MLIR by default in recent LLVM).
- Required libraries exist in the LLVM tree:
  - `MLIRTransformDialect`
  - `MLIRTransformDialectTransforms`
  - `MLIRLinalgTransformOps`

### 2) Link the required MLIR libraries in standalone-opt
- Update `standalone-opt/CMakeLists.txt` to link:
  - `MLIRTransformDialect`
  - `MLIRTransformDialectTransforms`
  - `MLIRLinalgTransformOps`
- These provide the Transform dialect itself, interpreter pass, and Linalg transform ops.

### 3) Register dialects and extensions in standalone-opt
- In `standalone-opt.cpp`, add:
  - `#include "mlir/Dialect/Transform/IR/TransformDialect.h"`
  - `#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"`
- In the registry:
  - `registry.insert<mlir::transform::TransformDialect>();`
  - `mlir::linalg::registerTransformDialectExtension(registry);`
- Optionally, use `registerAllDialects(registry);` and `registerAllExtensions(registry);` if you want every dialect/extension (more deps).

### 4) Decide how you will supply Transform IR
- **Embedded** (single file): put `transform.named_sequence @__transform_main` in the same module as payload. The module must have `transform.with_named_sequence` attribute. The interpreter pass can run directly on that file.
- **Separate library**: keep Transform IR in a separate file and load it with `-transform-preload-library` (set `-transform-library-paths`). Then run `-transform-interpreter` on the payload.

### 5) Create a minimal Transform schedule for your payload
- For your `linalg-matmul-fc-relu.mlir`, define a transform entry point that:
  - matches the `linalg.matmul` op,
  - tiles it (e.g., `transform.structured.tile_using_for`),
  - fuses its elementwise producers (e.g., `transform.structured.fuse` or `transform.structured.fuse_into_containing_op`),
  - returns `transform.yield`.

### 6) Run the interpreter pass in standalone-opt
- Example invocation (embedded schedule):
  - `standalone-opt -transform-interpreter linalg-matmul-fc-relu.mlir`
- If you use a separate library file:
  - `standalone-opt -transform-preload-library \
    -transform-library-paths=path/to/transform-library.mlir \
    -transform-interpreter linalg-matmul-fc-relu.mlir`

### 7) Optional safety/debug passes
- `-transform-dialect-check-uses` before interpreter to detect handle lifetime issues.
- `-transform-infer-effects` to infer side effects for named sequences.
- `-pass-pipeline="builtin.module(transform-dialect-check-uses,transform-interpreter)"` when you want explicit ordering.

## POC shape for linalg-matmul-fc-relu
- Payload: use your existing `mlir-standalone/test/Standalone/linalg-matmul-fc-relu.mlir` unchanged.
- Transform: create a small transform module that matches `linalg.matmul` and applies `structured.tile_using_for` + `structured.fuse`.
- Expected outcome: linalg loops/tiles + fused elementwise generics (depending on the exact transform ops and options you pick).

## Key takeaway
- Transform dialect = IR for transformation *specification*.
- Transform interpreter pass = the execution engine.
- To use Linalg transforms, you must register the **Linalg Transform extension** and link the **Linalg Transform Ops** library.
