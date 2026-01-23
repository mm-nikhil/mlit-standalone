func.func @fc_relu_tiled_fused(
  %A: tensor<MpxKpxf32>,     // lhs (maybe padded)
  %B: tensor<KpxNpxf32>,     // rhs (maybe padded)
  %bias: tensor<MpxNpxf32>  // bias (maybe padded)
) -> tensor<MpxNpxf32> {

  %c0 = arith.constant 0.0 : f32
  %out0 = tensor.empty() : tensor<MpxNpxf32>

  // ---- Tile sizes (starter) ----
  // We'll compute output C in tiles of (TM x TN).
  // Example: TM=64, TN=64: C_tile is 64x64 = 4096 floats = 16KB (fits L1 nicely).
  %TM = arith.constant 64 : index
  %TN = arith.constant 64 : index

  // K reduction tile: TK=64 (reduce in chunks so A/B panels are cache-friendly)
  %TK = arith.constant 64 : index

  // Outer loops over output tiles: these are PARALLEL dimensions (i and j).
  %out = scf.for %i = 0 to Mp step %TM iter_args(%C = %out0) -> tensor<MpxNpxf32> {
    %out2 = scf.for %j = 0 to Np step %TN iter_args(%C2 = %C) -> tensor<MpxNpxf32> {

      // 1) Extract the output tile we are going to compute: C_tile = C[i:i+TM, j:j+TN]
      %C_tile_init = tensor.extract_slice %C2[%i, %j] [%TM, %TN] [1, 1]
        : tensor<MpxNpxf32> to tensor<64x64xf32>

      // 2) REDUCTION over K happens here.
      //    Notice: iter_args(%Acc = %C_tile_init)
      //    This means: %Acc is the accumulator tile that gets updated each K-chunk.
      %C_tile = scf.for %k = 0 to Kp step %TK
                  iter_args(%Acc = %C_tile_init) -> tensor<64x64xf32> {

        // Extract A panel: A_panel = A[i:i+TM, k:k+TK]
        %A_panel = tensor.extract_slice %A[%i, %k] [%TM, %TK] [1, 1]
          : tensor<MpxKpxf32> to tensor<64x64xf32>

        // Extract B panel: B_panel = B[k:k+TK, j:j+TN]
        %B_panel = tensor.extract_slice %B[%k, %j] [%TK, %TN] [1, 1]
          : tensor<KpxNpxf32> to tensor<64x64xf32>

        // Compute partial matmul and ACCUMULATE into %Acc.
        // This is why K is reduction: multiple k-steps update the SAME C tile.
        %Acc_next = linalg.matmul
            ins(%A_panel, %B_panel : tensor<64x64xf32>, tensor<64x64xf32>)
            outs(%Acc : tensor<64x64xf32>) -> tensor<64x64xf32>

        // Yield updated accumulator back to the k-loop
        scf.yield %Acc_next : tensor<64x64xf32>
      }

      // 3) Fused epilogue: bias add + relu ON THE TILE
      %bias_tile = tensor.extract_slice %bias[%i, %j] [%TM, %TN] [1,1]
        : tensor<MpxNpxf32> to tensor<64x64xf32>

      %C_biased = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%C_tile, %bias_tile : tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%C_tile : tensor<64x64xf32>) -> tensor<64x64xf32>

      %C_relu = linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
        ins(%C_biased, %c0 : tensor<64x64xf32>, f32)
        outs(%C_biased : tensor<64x64xf32>) -> tensor<64x64xf32>

      // 4) Insert the finished tile back into the full output tensor
      %C3 = tensor.insert_slice %C_relu into %C2[%i, %j] [%TM, %TN] [1, 1]
        : tensor<64x64xf32> into tensor<MpxNpxf32>

      scf.yield %C3 : tensor<MpxNpxf32>
    }
    scf.yield %out2 : tensor<MpxNpxf32>
  }

  return %out : tensor<MpxNpxf32>
}
