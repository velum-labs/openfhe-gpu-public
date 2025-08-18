Troubleshooting
===============

Build issues
------------
- NVCC not found: set ``-DCMAKE_CUDA_COMPILER=/path/to/nvcc``
- Undefined references to ``CUDA::nvToolsExt``: ensure CUDA dev tools are installed
- SM architecture mismatch: adjust ``-arch`` in toolchain if needed (edit CMake if required)

Runtime issues
--------------
- ``cudaErrorInvalidValue``: check vector sizes are degree-multiples; ensure matching params between CPU and GPU
- ``out of memory``: reduce cached rotation keys, or N; call ``EnableMemoryPool`` to reuse buffers
- Wrong results: verify scaling factor/level alignment when adding; use provided helpers (``GetAdjustScalar``, ``Eval*ConstWithLoad``)

Diagnostics
-----------
- Enable NVTX ranges in ``CudaHelper.cu`` and profile with Nsight Systems
- Insert ``CudaHostSync()`` at phase boundaries to isolate errors (dev only)

