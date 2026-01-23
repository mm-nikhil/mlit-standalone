// This file embeds a Transform dialect schedule next to the payload IR.
// Running `standalone-opt -transform-interpreter` will interpret the transform
// sequence and rewrite the payload (tile + fuse) while leaving the transform IR
// in the output for inspection.
// RUN: standalone-opt -transform-interpreter %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

module attributes {transform.with_named_sequence} {

  transform.named_sequence @tile_policy_multilevel(
      %root: !transform.any_op {transform.readonly},
      %m_l2: !transform.param<i64> {transform.readonly},
      %n_l2: !transform.param<i64> {transform.readonly},
      %k_l2: !transform.param<i64> {transform.readonly},
      %m_l1: !transform.param<i64> {transform.readonly},
      %n_l1: !transform.param<i64> {transform.readonly},
      %k_l1: !transform.param<i64> {transform.readonly}) {

    // Match operations.
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %add = transform.structured.match ops{["linalg.elementwise"]}
      attributes{kind = #linalg.elementwise_kind<add>} in %root
        : (!transform.any_op) -> !transform.any_op
    %relu = transform.structured.match ops{["linalg.elementwise"]}
      attributes{kind = #linalg.elementwise_kind<max_signed>} in %root
        : (!transform.any_op) -> !transform.any_op

    // L2 tiling (outer tiles).
    %matmul_l2, %l2_m, %l2_n, %l2_k =
      transform.structured.tile_using_for %matmul tile_sizes [%m_l2, %n_l2, %k_l2]
        : (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
          -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // L1 tiling (inner tiles).
    %matmul_l1, %l1_m, %l1_n, %l1_k =
      transform.structured.tile_using_for %matmul_l2 tile_sizes [%m_l1, %n_l1, %k_l1]
        : (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
          -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Tile epilogue (M/N only, matching L2 tile size).
    %relu_tiled, %relu_m, %relu_n =
      transform.structured.tile_using_for %relu tile_sizes [%m_l2, %n_l2]
        : (!transform.any_op, !transform.param<i64>, !transform.param<i64>)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Fuse add into the relu loop.
    %add_fused, %loop_with_add =
      transform.structured.fuse_into_containing_op %add into %relu_m
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Cleanup.
    %func = transform.get_parent_op %relu_m : (!transform.any_op) -> !transform.any_op
    transform.apply_cse to %func : !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    transform.yield
  }

  transform.named_sequence @tile_policy_multilevel_x86(
      %root: !transform.any_op {transform.readonly}) {
    %m_l2 = transform.param.constant 128 : i64 -> !transform.param<i64>
    %n_l2 = transform.param.constant 128 : i64 -> !transform.param<i64>
    %k_l2 = transform.param.constant 256 : i64 -> !transform.param<i64>
    %m_l1 = transform.param.constant 32 : i64 -> !transform.param<i64>
    %n_l1 = transform.param.constant 32 : i64 -> !transform.param<i64>
    %k_l1 = transform.param.constant 32 : i64 -> !transform.param<i64>
    transform.include @tile_policy_multilevel failures(propagate)
      (%root, %m_l2, %n_l2, %k_l2, %m_l1, %n_l1, %k_l1)
        : (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>,
           !transform.param<i64>, !transform.param<i64>, !transform.param<i64>) -> ()
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.include @tile_policy_multilevel_x86 failures(propagate) (%root)
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
