// This file embeds a Transform dialect schedule next to the payload IR.
// Running `standalone-opt -transform-interpreter` will interpret the transform
// sequence and rewrite the payload (tile + fuse) while leaving the transform IR
// in the output for inspection.
// RUN: standalone-opt -transform-interpreter %s

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

  // Parameterized tiling policy: single-level M/N tiling + fusion, then K tiling.
  transform.named_sequence @tile_policy(
      %root: !transform.any_op {transform.readonly},
      %m_tile: !transform.param<i64> {transform.readonly},
      %n_tile: !transform.param<i64> {transform.readonly},
      %k_tile: !transform.param<i64> {transform.readonly}) {
    // 1) Match the specific ops we want to transform.
    //    We avoid consumer-chain indexing so the schedule is robust if extra
    //    uses are added in the future.
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %add = transform.structured.match ops{["linalg.elementwise"]}
      attributes{kind = #linalg.elementwise_kind<add>} in %root
        : (!transform.any_op) -> !transform.any_op
    %relu = transform.structured.match ops{["linalg.elementwise"]}
      attributes{kind = #linalg.elementwise_kind<max_signed>} in %root
        : (!transform.any_op) -> !transform.any_op

    // 2) Tile the consumer (ReLU) on its parallel dimensions.
    //    This gives us (i,j) tiles for the output C and is the right place to
    //    fuse the epilogue.
    %relu_tiled, %loop =
      transform.structured.tile_using_forall %relu tile_sizes [%m_tile, %n_tile]
        : (!transform.any_op, !transform.param<i64>, !transform.param<i64>)
          -> (!transform.any_op, !transform.any_op)
    // 3) Fuse producers into the tiled loop in producer-to-consumer order.
    //    Fusing add before matmul preserves def-use ordering inside the loop.
    %add_fused, %loop_with_add =
      transform.structured.fuse_into_containing_op %add into %loop
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %matmul_fused, %loop_with_matmul =
      transform.structured.fuse_into_containing_op %matmul into %loop_with_add
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // 4) Tile the reduction dimension K within the fused matmul.
    //    This improves locality for A/B panels while keeping the epilogue
    //    outside the K loop.
    %matmul_k, %k_loop = transform.structured.tile_using_for %matmul_fused
      tile_sizes [0, 0, %k_tile]
        : (!transform.any_op, !transform.param<i64>)
          -> (!transform.any_op, !transform.any_op)

    // 5) Cleanup on the surrounding function (not the module with transform IR).
    %func = transform.get_parent_op %loop_with_matmul : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.yield
  }

  // Tiling + K-tiling + vectorization (targeting the tiled matmul and relu).
  transform.named_sequence @tile_policy_vec(
      %root: !transform.any_op {transform.readonly},
      %m_tile: !transform.param<i64> {transform.readonly},
      %n_tile: !transform.param<i64> {transform.readonly},
      %k_tile: !transform.param<i64> {transform.readonly}) {
    // Same structure as @tile_policy, with vectorization after tiling.
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %add = transform.structured.match ops{["linalg.elementwise"]}
      attributes{kind = #linalg.elementwise_kind<add>} in %root
        : (!transform.any_op) -> !transform.any_op
    %relu = transform.structured.match ops{["linalg.elementwise"]}
      attributes{kind = #linalg.elementwise_kind<max_signed>} in %root
        : (!transform.any_op) -> !transform.any_op

    %relu_tiled, %loop =
      transform.structured.tile_using_forall %relu tile_sizes [%m_tile, %n_tile]
        : (!transform.any_op, !transform.param<i64>, !transform.param<i64>)
          -> (!transform.any_op, !transform.any_op)
    %add_fused, %loop_with_add =
      transform.structured.fuse_into_containing_op %add into %loop
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %matmul_fused, %loop_with_matmul =
      transform.structured.fuse_into_containing_op %matmul into %loop_with_add
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %matmul_k, %k_loop = transform.structured.tile_using_for %matmul_fused
      tile_sizes [0, 0, %k_tile]
        : (!transform.any_op, !transform.param<i64>)
          -> (!transform.any_op, !transform.any_op)

    %func = transform.get_parent_op %loop_with_matmul : (!transform.any_op) -> !transform.any_op

    // Vectorize the tiled matmul and the tiled epilogue.
    transform.structured.vectorize %matmul_k : !transform.any_op
    transform.structured.vectorize %relu_tiled : !transform.any_op

    transform.apply_cse to %func : !transform.any_op
    transform.yield
  }

  // x86 defaults for the local i5-13420H (AVX2): 64x64 tiles with K=32.
  transform.named_sequence @tile_policy_x86(%root: !transform.any_op {transform.readonly}) {
    %m_tile = transform.param.constant 64 : i64 -> !transform.param<i64>
    %n_tile = transform.param.constant 64 : i64 -> !transform.param<i64>
    %k_tile = transform.param.constant 32 : i64 -> !transform.param<i64>
    transform.include @tile_policy failures(propagate)
      (%root, %m_tile, %n_tile, %k_tile)
        : (!transform.any_op, !transform.param<i64>, !transform.param<i64>,
           !transform.param<i64>) -> ()
    transform.yield
  }

  // This is the default entry point that the interpreter looks for.
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.include @tile_policy_x86 failures(propagate) (%root)
      : (!transform.any_op) -> ()
    transform.yield
  }

  // --- Payload IR ---------------------------------------------------------
  // Payload copied from matmul-fc-relu.mlir (elementwise ops).
  func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                     %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                     -> tensor<512x512xf32> {
    // Matrix-matrix multiplication.
    %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                            outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

    // Elementwise addition.
    %biased = linalg.elementwise kind=#linalg.elementwise_kind<add>
      ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
      outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

    // Elementwise max with 0 (ReLU).
    %c0f = arith.constant 0.0 : f32
    %relued = linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>]
      ins(%biased, %c0f : tensor<512x512xf32>, f32)
      outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
    func.return %relued : tensor<512x512xf32>
  }

  func.func @main() {
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1.0 : f32
    %c2 = arith.constant 2.0 : f32
    %lhs0 = tensor.empty() : tensor<512x512xf32>
    %rhs0 = tensor.empty() : tensor<512x512xf32>
    %bias0 = tensor.empty() : tensor<512x512xf32>
    %out0 = tensor.empty() : tensor<512x512xf32>
    %lhs = linalg.fill ins(%c1 : f32) outs(%lhs0 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %rhs = linalg.fill ins(%c2 : f32) outs(%rhs0 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %bias = linalg.fill ins(%c0 : f32) outs(%bias0 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %out = linalg.fill ins(%c0 : f32) outs(%out0 : tensor<512x512xf32>) -> tensor<512x512xf32>
    %res = call @fc_relu(%lhs, %rhs, %bias, %out)
      : (tensor<512x512xf32>, tensor<512x512xf32>, tensor<512x512xf32>,
         tensor<512x512xf32>) -> tensor<512x512xf32>
    func.return
  }
}
