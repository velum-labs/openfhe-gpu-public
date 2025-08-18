Installation and Build
======================

Requirements
------------
- NVIDIA GPU with CUDA capability (sm_70+ recommended)
- CUDA Toolkit 12.x (adjust path if different)
- CMake ≥ 3.18
- g++ ≥ 9 or clang++ ≥ 10

Configure
---------

The top-level CMake explicitly sets ``CMAKE_CUDA_COMPILER`` and links CUDA runtime and NVTX in core/pke. Ensure ``/usr/local/cuda`` exists or override path:

.. code-block:: bash

  mkdir -p build && cd build
  cmake -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DBUILD_EXAMPLES=ON -DBUILD_UNITTESTS=ON ..

Build
-----

.. code-block:: bash

  make -j$(nproc)

Verify
------

- GPU examples will be placed under ``build/bin/examples/pke`` (e.g., ``advanced-ckks-bootstrapping-gpu``).
- Run unit tests (if enabled): ``build/unittest/pke_tests`` and ensure GPU tests pass on your hardware.

Common pitfalls
---------------
- NVCC path mismatch: set ``-DCMAKE_CUDA_COMPILER``.
- Driver/runtime mismatch: verify ``nvidia-smi`` and ``nvcc --version`` compatibility.
- Missing NVTX: ensure CUDA samples/devtools installed; or disable profiling in code where compiled.

