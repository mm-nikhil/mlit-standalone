// This file embeds a Transform dialect schedule next to the payload IR.
// Running `standalone-opt -transform-interpreter` will interpret the transform
// sequence and rewrite the payload (tile + fuse) while leaving the transform IR
// in the output for inspection.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

module attributes {transform.with_named_sequence} {
  // Match a single linalg.matmul op. The matcher is re-usable and keeps the
  // main sequence cleaner.
  transform.named_sequence @match_matmul(%op: !transform.any_op {transform.readonly})
      -> !transform.any_op {
    transform.match.operation_name %op ["linalg.matmul"]
      : !transform.any_op
    transform.yield %op : !transform.any_op
  }

  // This is the default entry point that the interpreter looks for.
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    // Collect all matmul ops under the root (our module). In this file there is
    // only one, but collect_matching scales to multiple matches.
    %matmul = transform.collect_matching @match_matmul in %root
      : (!transform.any_op) -> !transform.any_op

    // Walk forward to the elementwise consumer chain:
    //   matmul -> elementwise add -> elementwise relu
    // This assumes there is exactly one consumer at each step.
    %add = transform.get_consumers_of_result %matmul[0]
      : (!transform.any_op) -> !transform.any_op
    %relu = transform.get_consumers_of_result %add[0]
      : (!transform.any_op) -> !transform.any_op

    // Tile the consumer (ReLU) and fuse its producers (add + matmul).
    // Tile sizes [64, 64] mean 64x64 tiles over the two parallel dims of the
    // elementwise op. This is a common starting point for cache locality.
    %fused, %i, %j = transform.structured.fuse %relu tile_sizes [64, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Optional cleanup: apply CSE to the surrounding function, not the module
    // that contains the transform IR, to avoid transforming the transform ops.
    %func = transform.get_parent_op %fused : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.yield
  }

  // --- Payload IR ---------------------------------------------------------
  // NOTE: We keep this payload in terms of linalg.matmul + linalg.generic to
  // ensure tiling+fusion runs reliably. The transform schedule is the same as
  // for linalg.elementwise, but generic is a stable baseline for transforms.
  func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                     %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                     -> tensor<512x512xf32> {
    // Matrix-matrix multiplication.
    %matmul = linalg.matmul ins(%lhs, %rhs : tensor<512x512xf32>, tensor<512x512xf32>)
                            outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

    // Elementwise addition via linalg.generic.
    %biased = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]
      } ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
        outs(%output : tensor<512x512xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
    } -> tensor<512x512xf32>

    // Elementwise max with 0 (ReLU) via linalg.generic.
    %c0f = arith.constant 0.0 : f32
    %relued = linalg.generic {
        indexing_maps = [#map, #map1, #map],
        iterator_types = ["parallel", "parallel"]
      } ins(%biased, %c0f : tensor<512x512xf32>, f32)
        outs(%output : tensor<512x512xf32>) {
      ^bb0(%a: f32, %b: f32, %c: f32):
        %max = arith.maximumf %a, %b : f32
        linalg.yield %max : f32
    } -> tensor<512x512xf32>
    func.return %relued : tensor<512x512xf32>
  }
}
