GPU Quickstart
==============

This guide shows how to build and run CKKS GPU-accelerated examples.

Prerequisites
-------------

- CUDA Toolkit installed (ensure `nvcc` is available). The build expects `/usr/local/cuda`; adjust `CMAKE_CUDA_COMPILER` if needed.
- C++17 compiler (g++ >= 9 or clang++ >= 10).

Build
-----

```
mkdir build && cd build
cmake .. -DBUILD_SHARED=ON -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
make -j
```

Run Examples
------------

The following examples exercise GPU paths:

- `advanced-ckks-bootstrapping-gpu`
- `ckks-gpu-bookkeeping`
- `ckks-gpu-keyswitch`

```
./bin/examples/pke/advanced-ckks-bootstrapping-gpu
```

Tips
----

- If CUDA is installed in a non-default location, pass `-DCMAKE_CUDA_COMPILER=/path/to/nvcc` to CMake.
- To speed up allocations during heavy workloads, enable the RMM memory pool via `ckks::MemoryPool`.
- Use `nvidia-smi` to monitor GPU utilization and memory.

Links
-----

- See the :doc:`../modules/core/gpu/architecture` and :doc:`../modules/core/gpu/api` pages for deeper technical details.

