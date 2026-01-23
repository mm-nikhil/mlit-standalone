// Re-usable matcher: take an op handle and assert it is a linalg.matmul.
// Returns the SAME handle (now verified) so it can be re-used as a “filter”.
transform.named_sequence @match_matmul(%op: !transform.any_op {transform.readonly})
    -> !transform.any_op {
  // Checks operation name == "linalg.matmul".
  // If it doesn't match, this emits a (silenceable) failure.
  transform.match.operation_name %op ["linalg.matmul"] : !transform.any_op

  // Return the (verified) handle.
  transform.yield %op : !transform.any_op
}

transform.named_sequence @tile_policy(
    %root: !transform.any_op {transform.readonly},
    %m_tile: !transform.param<i64> {transform.readonly},
    %n_tile: !transform.param<i64> {transform.readonly},
    %k_tile: !transform.param<i64> {transform.readonly}) {

  // 1) Collect all matmuls under %root.
  // IMPORTANT: this handle may be associated with *a list* of ops.
  %matmul = transform.collect_matching @match_matmul in %root
    : (!transform.any_op) -> !transform.any_op

  // 2) Walk forward through single-consumer chain.
  // %matmul[0] means: "take the first op in the handle’s op-list".
  // This silently bakes in: "there is exactly one matmul we care about".
  %add = transform.get_consumers_of_result %matmul[0]
    : (!transform.any_op) -> !transform.any_op

  // Same assumption: exactly one consumer of add-result (the relu).
  %relu = transform.get_consumers_of_result %add[0]
    : (!transform.any_op) -> !transform.any_op

  // Debug prints: useful while developing. Fine.
  transform.debug.emit_remark_at %matmul, "tile_policy: matmul" : !transform.any_op
  transform.debug.emit_remark_at %add, "tile_policy: add" : !transform.any_op
  transform.debug.emit_remark_at %relu, "tile_policy: relu" : !transform.any_op

  transform.debug.emit_param_as_remark %m_tile, "tile m" at %relu
    : !transform.param<i64>, !transform.any_op
  transform.debug.emit_param_as_remark %n_tile, "tile n" at %relu
    : !transform.param<i64>, !transform.any_op
  transform.debug.emit_param_as_remark %k_tile, "tile k" at %relu
    : !transform.param<i64>, !transform.any_op

  // 3) Tile+fuse on the CONSUMER (relu).
  //
  // Semantics: create loops for the relu iteration space (M,N),
  // then "pull in" producer computations needed for each tile.
  //
  // This is the structured-fusion trick the tutorial highlights:
  // tile the last op, then fuse producers into the loop nest. :contentReference[oaicite:2]{index=2}
  //
  // Returned handles:
  //   %relu_tiled  -> the tiled relu op (new payload op)
  //   %tile_i/%tile_j -> handles to the generated loops (or loop-like containers),
  //                      depending on the implementation of structured.fuse.
  %relu_tiled, %tile_i, %tile_j =
    transform.structured.fuse %relu tile_sizes [%m_tile, %n_tile]
      : (!transform.any_op, !transform.param<i64>, !transform.param<i64>)
        -> (!transform.any_op, !transform.any_op, !transform.any_op)

  // 4) Find the matmul *inside* the tiled region and tile K there.
  //
  // You are currently searching inside %tile_j specifically.
  // That assumes:
  //   - %tile_j is a container op that actually *contains* the fused matmul
  //   - and the matmul ended up nested under that particular handle
  //
  // In practice, it’s safer to search in the tiled relu op or in the outer loop,
  // because “which loop handle contains what” can change as implementations evolve.
  %matmul_inner = transform.collect_matching @match_matmul in %tile_j
    : (!transform.any_op) -> !transform.any_op

  // Tile the reduction dimension (K).
  // tile_sizes [0, 0, %k_tile] means:
  //   - do not tile M (i)
  //   - do not tile N (j)
  //   - split/tile K by %k_tile
  //
  // Implementation typically produces an scf.for over k blocks and threads an
  // accumulator via iter_args/yield (that’s the "reduction in K"). 
  %matmul_k, %k_loop =
    transform.structured.tile_using_for %matmul_inner tile_sizes [0, 0, %k_tile]
      : (!transform.any_op, !transform.param<i64>)
        -> (!transform.any_op, !transform.any_op)

  transform.debug.emit_remark_at %matmul_k, "tile_policy: matmul k-tiled"
    : !transform.any_op

  // 5) Cleanup: CSE on the function containing the transformed ops.
  // Good instinct: don’t run CSE on the module containing transform IR.
  %func = transform.get_parent_op %matmul_k : (!transform.any_op) -> !transform.any_op
  transform.apply_cse to %func : !transform.any_op

  transform.yield
}
