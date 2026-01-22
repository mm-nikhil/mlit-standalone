#!/usr/bin/env python3
"""Run LLVM lit in a single process (no multiprocessing).

This is a workaround for environments where Python's multiprocessing
(semaphores) is not permitted. It executes tests serially while preserving
lit's normal command-line interface and output formatting.
"""

from __future__ import annotations

import os
import sys
import time


def _infer_llvm_binary_dir() -> str:
    # Allow callers to override explicitly.
    env_dir = os.environ.get("LLVM_BINARY_DIR")
    if env_dir:
        return os.path.abspath(env_dir)

    # Default to the common layout in this workspace.
    # If your LLVM build lives elsewhere, set LLVM_BINARY_DIR in the env.
    default_dir = "/home/ws-mm-69/workspace/llvm-project/build"
    return os.path.abspath(default_dir)


def _setup_lit_on_syspath() -> None:
    llvm_binary_dir = _infer_llvm_binary_dir()
    llvm_source_dir = os.path.join(os.path.dirname(llvm_binary_dir), "llvm")
    lit_dir = os.path.join(llvm_source_dir, "utils", "lit")
    if not os.path.isdir(lit_dir):
        sys.stderr.write(
            "error: cannot find lit sources at '{}'\n".format(lit_dir)
        )
        sys.stderr.write(
            "Set LLVM_BINARY_DIR to your LLVM build directory.\n"
        )
        sys.exit(2)
    sys.path.insert(0, lit_dir)


# Monkeypatch lit to avoid multiprocessing.

def _patch_lit_for_serial() -> None:
    import lit.run
    import lit.worker

    def _execute_serial(self, deadline):
        self._increase_process_limit()

        for test in self.tests:
            if time.time() > deadline:
                raise lit.run.TimeoutError()

            # Execute directly in-process using the configured test runner.
            result = lit.worker._execute(test, self.lit_config)
            test.setResult(result)

            # Update progress output and failure accounting.
            self.progress_callback(test)
            if test.isFailure():
                self.failures += 1
                if self.failures == self.max_failures:
                    raise lit.run.MaxFailuresError()

    lit.run.Run._execute = _execute_serial


def main() -> int:
    _setup_lit_on_syspath()
    _patch_lit_for_serial()

    from lit.main import main as lit_main

    # Minimal builtin parameters; this keeps CLI compatibility with llvm-lit.
    builtin_parameters = {"build_mode": ".", "config_map": {}}
    return lit_main(builtin_parameters)


if __name__ == "__main__":
    raise SystemExit(main())
