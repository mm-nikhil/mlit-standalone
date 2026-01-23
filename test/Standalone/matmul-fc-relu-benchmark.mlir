// RUN: standalone-opt %s
// Baseline (untiled) payload + simple main for CPU runner.

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
