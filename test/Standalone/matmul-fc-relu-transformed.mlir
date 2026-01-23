// RUN: standalone-opt -transform-interpreter %s

#map  = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

module attributes {transform.with_named_sequence} {

  // --- Utility: match the function we care about, by symbol name.
  transform.named_sequence @match_fc_relu(%root: !transform.any_op {transform.readonly})
      -> !transform.any_op {
    %funcs = transform.structured.match ops{["func.func"]}
      attributes{sym_name = "fc_relu"} in %root
        : (!transform.any_op) -> !transform.any_op
    // We expect exactly one.
    transform.yield %funcs : !transform.any_op
  }

  // Entry point expected by your benchmark.py for 6-arg tiles:
  //   --tilesize m2,n2,k2,m1,n1,k1
  // -> transform-interpreter{entry-point=tile_policy_multilevel debug-bind-trailing-args=...}
  transform.named_sequence @tile_policy_multilevel(
      %root: !transform.any_op {transform.readonly},
      %m2: !transform.param<i64> {transform.readonly},
      %n2: !transform.param<i64> {transform.readonly},
      %k2: !transform.param<i64> {transform.readonly},
      %m1: !transform.param<i64> {transform.readonly},
      %n1: !transform.param<i64> {transform.readonly},
      %k1: !transform.param<i64> {transform.readonly}) {

    // 0) Scope everything to @fc_relu so “match” is not global.
    %func = transform.include @match_fc_relu failures(propagate) (%root)
      : (!transform.any_op) -> !transform.any_op

    // 1) Match sink relu inside fc_relu.
    %relu = transform.structured.match ops{["linalg.elementwise"]}
      attributes{kind = #linalg.elementwise_kind<max_signed>} in %func
        : (!transform.any_op) -> !transform.any_op

    // 2) Walk producers along actual dataflow (no consumer ordering games).
    %add = transform.get_producer_of_operand %relu[0]
      : (!transform.any_op) -> !transform.any_op
    %matmul = transform.get_producer_of_operand %add[0]
      : (!transform.any_op) -> !transform.any_op

    transform.match.operation_name %add ["linalg.elementwise"] : !transform.any_op
    transform.match.operation_name %matmul ["linalg.matmul"] : !transform.any_op

    // OUTER TILE SPINE: tile the CONSUMER (relu) by (m2, n2) using forall.
    // This is the “roof”: everything fuses into this one nest.
    // NOTE: On some LLVM snapshots, the result order is (%forall, %tiled).
    // If you get a type mismatch, swap the LHS order.
    %relu_tiled, %forall =
      transform.structured.tile_using_forall %relu tile_sizes [%m2, %n2]
        : (!transform.any_op, !transform.param<i64>, !transform.param<i64>)
          -> (!transform.any_op, !transform.any_op)

    // Fuse producers INTO the forall nest.
    %add_fused, %forall2 =
      transform.structured.fuse_into_containing_op %add into %forall
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %matmul_fused, %forall3 =
      transform.structured.fuse_into_containing_op %matmul into %forall2
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // REDUCTION BLOCKING: strip-mine K by k2 INSIDE the fused matmul.
    // This is where your “k2=8” magic comes from: tiny K chunks improve locality.
    %matmul_k2, %k2_loop =
      transform.structured.tile_using_for %matmul_fused tile_sizes [0, 0, %k2]
        : (!transform.any_op, !transform.param<i64>)
          -> (!transform.any_op, !transform.any_op)

    // INNER TILE: now tile the already-k2-stripmined matmul by (m1, n1, k1).
    // This gives you the classic nested (M,N,K) block inside each outer tile.
    %matmul_l1, %m1_loop, %n1_loop, %k1_loop =
      transform.structured.tile_using_for %matmul_k2 tile_sizes [%m1, %n1, %k1]
        : (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>)
          -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Cleanup only within fc_relu.
    transform.apply_cse to %func : !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op

    transform.yield
  }

  // Default entry point if you don’t pass --tilesize.
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %m2 = transform.param.constant 256 : i64 -> !transform.param<i64>
    %n2 = transform.param.constant 256 : i64 -> !transform.param<i64>
    %k2 = transform.param.constant 8   : i64 -> !transform.param<i64> // mirrors your best case
    %m1 = transform.param.constant 64  : i64 -> !transform.param<i64>
    %n1 = transform.param.constant 64  : i64 -> !transform.param<i64>
    %k1 = transform.param.constant 64  : i64 -> !transform.param<i64>
    transform.include @tile_policy_multilevel failures(propagate)
      (%root, %m2, %n2, %k2, %m1, %n1, %k1)
        : (!transform.any_op, !transform.param<i64>, !transform.param<i64>, !transform.param<i64>,
           !transform.param<i64>, !transform.param<i64>, !transform.param<i64>) -> ()
    transform.yield
  }

  // --- Payload IR ---------------------------------------------------------
  func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                     %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                     -> tensor<512x512xf32> {
    %matmul = linalg.matmul
      ins(%lhs, %rhs : tensor<512x512xf32>, tensor<512x512xf32>)
      outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

    %biased = linalg.elementwise kind=#linalg.elementwise_kind<add>
      ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
      outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

    %c0f = arith.constant 0.0 : f32

    // scalar operand uses map -> ()
    %relued = linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
      indexing_maps = [#map, #map1, #map]
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
    %out  = linalg.fill ins(%c0 : f32) outs(%out0  : tensor<512x512xf32>) -> tensor<512x512xf32>
    %res = call @fc_relu(%lhs, %rhs, %bias, %out)
      : (tensor<512x512xf32>, tensor<512x512xf32>, tensor<512x512xf32>, tensor<512x512xf32>)
        -> tensor<512x512xf32>
    func.return
  }
}
