// RUN: standalone-opt %s | FileCheck %s

module {
  // CHECK-LABEL: func @matmul
  func.func @matmul(%a: tensor<1024x1024xf32>, %b: tensor<1024x1024xf32>,
                    %c: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> {
    // CHECK: linalg.matmul
    %0 = linalg.matmul ins(%a, %b : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                      outs(%c : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %0 : tensor<1024x1024xf32>
  }
}
