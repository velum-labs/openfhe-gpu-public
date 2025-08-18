Realistic Examples
==================

1) Privacy-preserving model scoring (CKKS)
------------------------------------------
Goal: evaluate a polynomial approximation of a logistic/activation over encrypted features.

Steps:
- Build a CKKS ``CryptoContext`` with depth for your polynomial (degree 5–15 typical).
- Encrypt batched feature vectors into one ciphertext.
- Move to GPU: ``ckks::Context gpu = GenGPUContext(cryptoParams)``; load eval mult/relin keys.
- Use GPU ``EvalChebyshevSeriesGPU`` to evaluate the polynomial; use fused rescale to manage depth.
- Return to CPU and decrypt the scores.

Reference code: ``src/pke/lib/scheme/ckksrns/ckksrns-advancedshe-gpu.cpp`` and ``src/pke/examples/ckks-gpu-bookkeeping.cpp``.

2) Fast CKKS bootstrapping for analytics pipelines
--------------------------------------------------
Goal: refresh ciphertexts to regain levels mid-pipeline to continue analytics.

Steps:
- Enable FHE and run ``EvalBootstrapSetup`` to precompute matrices.
- Preload a subset/all rotation keys into GPU (see ``advanced-ckks-bootstrapping-gpu.cpp``).
- Call ``EvalBootstrapGPU``; monitor time for CoeffsToSlots, approx mod reduction, SlotsToCoeffs.
- Resume downstream GPU/CPU operations.

3) Vector rotations for encrypted convolutions
----------------------------------------------
Goal: rotate packed vectors and aggregate windows (encrypted sliding window).

Steps:
- Generate rotation keys for required shifts.
- Preload keys into ``gpu_context.preloaded_rotation_key_map``.
- Use ``EvalAtIndex`` and ``AutomorphismTransformInPlace`` to shift and sum.

4) Key switching microservice
-----------------------------
Goal: offload CKKS key switching to a GPU worker.

Steps:
- Accept a raised digit bundle (ModUp output) and a rotation/relin key id.
- Keep keys resident on GPU; call ``KeySwitch`` with fused inner-product kernel.
- Return (ax,bx) switched limbs; caller performs ModDown/NTT on its side or via API.

End-to-end scripts
------------------
- Build and run: ``build/bin/examples/pke/advanced-ckks-bootstrapping-gpu``
- Unit-style flows: ``src/pke/examples/ckks-gpu-keyswitch.cpp`` and ``ckks-gpu-bookkeeping.cpp``

