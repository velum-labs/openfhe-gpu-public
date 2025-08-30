GPU Architecture Overview
==========================

This section describes the GPU acceleration components integrated into the core library for CKKS operations, focusing on CUDA-based execution and device memory management.

High-level Design
-----------------

- **CUDA kernels**: Core arithmetic and NTT operations run on the GPU (see `src/core/lib/gpu/Context.cu`, `src/core/lib/gpu/NttImple.cu`).
- **Device containers**: `ckks::DeviceVector` wraps device memory using RMM `device_uvector` for efficient allocation and async copies.
- **Context and keys**: `ckks::Context` orchestrates GPU operations, while `ckks::EvaluationKey` contains relinearization data on device.
- **Memory pool**: `ckks::MemoryPool` configures per-device memory bins via RMM to reduce allocation overhead.
Key Components
--------------

- `ckks::DeviceVector` (`src/core/lib/gpu/DeviceVector.h`) wraps a device array of 64-bit words with copy construction from host vectors, implicit host conversion, append, resize, and constant fill operations. It uses `cudaStreamLegacy` for async transfers.
- `ckks::Context` (`src/core/lib/gpu/Context.h`) exposes high-level CKKS operations on ciphertexts and plaintexts: multiplication, relinearization, squaring, automorphisms/rotations, add/sub/const, rescale, key switching, and monomial multiplication. It maintains parameters, NTT tables, and kernel metadata.
- `ckks::MemoryPool` (`src/core/lib/gpu/MemoryPool.h`) sets up RMM binning memory resource with heuristic bin sizes derived from CKKS parameters to amortize allocations during bootstrapping and heavy ops.
- Utility helpers (`src/core/lib/gpu/Utils.h`) translate between OpenFHE CPU types (`DCRTPoly`, `Ciphertext`) and GPU types (`ckks::DeviceVector`, `ckks::Ciphertext`, `ckks::CtAccurate`). They also build a GPU `Context` from `CryptoParams` via `GenGPUContext`.

Build and Dependencies
----------------------

- CUDA 11+ is required; the project sets `CMAKE_CUDA_COMPILER` to `/usr/local/cuda/bin/nvcc` by default. Adjust if your CUDA path differs.
- Third-party dependencies include Thrust and NVIDIA RMM. CMake targets link to `Thrust`, `CUDA::cudart`, `CUDA::nvToolsExt`, and `rmm` in `src/pke/CMakeLists.txt` and `src/core/CMakeLists.txt`.
- Library targets also add `/usr/local/cuda/lib64` to link directories.

Runtime Considerations
----------------------

- Asynchronous copies are used for host/device transfers; synchronize using `CudaHostSync()` when needed (see `Define.h`).
- Use `ckks::MemoryPool::UseMemoryPool(true)` to enable the RMM memory pool for the current device during heavy workloads.
- For tracing, `CudaNvtxStart/Stop` can annotate GPU regions when `nvToolsExt` is available.

Related Files
-------------

- `src/core/lib/gpu/Context.cu`
- `src/core/lib/gpu/NttImple.cu`
- `src/core/lib/gpu/DeviceVector.h`
- `src/core/lib/gpu/MemoryPool.h`
- `src/core/lib/gpu/Utils.h`
