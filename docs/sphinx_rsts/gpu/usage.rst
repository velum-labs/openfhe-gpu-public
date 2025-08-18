Using the GPU APIs
==================

High-level flow
---------------
1) Construct CPU CryptoContext (CKKS) and keys as usual.
2) Create a GPU ``ckks::Context`` using ``GenGPUContext(cryptoParams)``.
3) Optionally preload evaluation and rotation keys to GPU.
4) Convert CPU ciphertexts/plaintexts to GPU via loaders in ``gpu/Utils.h`` (e.g., ``LoadAccurateCiphertext``).
5) Invoke GPU operations on ``ckks::Context`` (EvalMult/EvalSquare/KeySwitch/Rescale/etc.).
6) Convert results back to CPU when needed.

Key helpers
-----------
- ``GenGPUContext(cryptoParams)``: builds ``ckks::Context`` with RNS params, primes, roots
- ``LoadEvalMultRelinKey(cc [, keyTag])`` and ``LoadRelinKey(evalKey)``
- ``LoadAccurateCiphertext(ct)`` / ``LoadCiphertext(ct)``
- ``loadIntoDeviceVector(DCRTPoly)`` and ``loadIntoDCRTPoly(DeviceVector, params)``

Core methods (subset)
---------------------
- ``EvalMultAndRelin(ct1, ct2, evk)``
- ``EvalSquareAndRelin(ct, evk)``
- ``ModUp(vec)`` / ``ModDown(vec, out)`` / ``Rescale(ct)`` / ``DropLimbs(ct, n)``
- ``KeySwitch(raisedDigits, evk, out_ax, out_bx)``
- ``AutomorphismTransformInPlace(ct, auto_index or index_map)``
- ``EvalMultConst(InPlace)`` and scalar add/sub helpers with on-device scaling

Bootstrapping
-------------
``CryptoContext::EvalBootstrapGPU(ciph, gpu_context)`` offloads CoeffsToSlots, approx mod reduction, and SlotsToCoeffs. Preload keys to minimize PCIe transfers.

Threading/streams
-----------------
Kernels use default stream; memory pool can be enabled via ``Context::EnableMemoryPool()``. Multi-GPU is not implemented in this branch.

