# An out-of-tree dialect template for MLIR

This repository contains a template for an out-of-tree [MLIR](https://mlir.llvm.org/) dialect as well as a
standalone `opt`-like tool to operate on that dialect.

## How to build

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
cmake --build . --target check-standalone-opt
```
To build the documentation from the TableGen description of the dialect
operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with
CMake so that it installs `FileCheck` to the chosen installation prefix.

## License

This dialect template is made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.

## MLIR-Build

cmake -G Ninja -S ~/workspace/mlir-standalone -B ~/workspace/mlir-standalone/build \
  -DMLIR_DIR=~/workspace/llvm-project/build/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=~/workspace/llvm-project/build/bin/llvm-lit

cmake --build ~/workspace/mlir-standalone/build
ninja -j $(nproc)

## Transform Interpreter

To apply a Transform dialect schedule embedded in a `.mlir` file and dump the
resulting payload IR, run the transform interpreter pass:

```sh
./build/bin/standalone-opt -transform-interpreter test/Standalone/matmul-fc-relu-transformed.mlir \
  > build/transformed.mlir
```

This invokes the `transform-interpreter` pass (default entry point
`__transform_main`) and prints the rewritten IR.
