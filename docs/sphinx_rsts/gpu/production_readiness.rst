Production Readiness Audit
==========================

Summary
-------
The CUDA path is functional for CKKS and bootstrapping but labeled experimental. Several areas need hardening before declaring GA.

Build and portability
---------------------
- CMake hardcodes ``CMAKE_CUDA_COMPILER`` to ``/usr/local/cuda/bin/nvcc`` in top-level; make configurable via option and detect CUDA toolkit with ``find_package(CUDAToolkit)``
- Linker paths include ``/usr/local/cuda-12.3/lib64``; replace with toolkit variables to avoid version pinning
- Add CI jobs with GPU runners (or device emulation) to validate builds and examples

APIs and ergonomics
-------------------
- Provide clear public headers for GPU helpers under a stable include path (``openfhe/gpu``)
- Ensure functions accept const-correct spans; avoid raw pointers in public APIs where feasible
- Document lifetime/ownership of preloaded key maps and memory pool

Correctness/safety
------------------
- Add cross-check tests: compare CPU vs GPU for representative operations (EvalMult, KeySwitch, rotations, bootstrapping) across multiple N and level budgets
- Validate scaling-factor and level adjustments systematically; encapsulate “adjust + rescale” patterns
- Enhance argument validation in kernels and host stubs; return status and propagate errors

Performance and memory
----------------------
- Key cache policy: configurable LRU of rotation keys; preloading strategies based on bootstrapping plans
- Expose memory pool statistics; free/reuse buffers between phases
- Explore stream parallelism for ModUp/KeySwitch batches

Security
--------
- Ensure no sensitive host/GPU buffers are left uncleared in production builds (optional zeroization mode)
- Review side-channel surfaces (timing uniformity where applicable)

Licensing
---------
- GPU code carries CC BY-NC-SA 4.0; clarify scope and add gating to disable GPU in commercial builds by default

Documentation
-------------
- This section provides install, usage, examples, and troubleshooting. Add API reference snippets and code breadcrumbs to source files.

Roadmap checklist
-----------------
- [ ] Replace hardcoded CUDA paths with ``find_package(CUDAToolkit)`` integration
- [ ] Introduce ``OPENFHE_ENABLE_CUDA`` option; build guards around GPU files
- [ ] Add GPU CI builds and regression tests against CPU
- [ ] Implement rotation key LRU cache and metrics
- [ ] Harden error handling and add unit tests for edge-cases
- [ ] Clarify/rework licensing for production

