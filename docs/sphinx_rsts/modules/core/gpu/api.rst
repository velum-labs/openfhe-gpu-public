GPU API Reference (C++)
=======================

This reference catalogs the primary GPU-facing C++ types under `src/core/lib/gpu` and their most relevant methods for CKKS acceleration.

Namespaces and Typedefs
-----------------------

- **Namespace**: `ckks`
- **Types**:
  - `using word64 = uint64_t;`
  - `using word128 = __uint128_t;`
  - `using HostVector = thrust::host_vector<word64>;`
  - `class DeviceVector : public rmm::device_uvector<word64>`

DeviceVector
------------

Header: `src/core/lib/gpu/DeviceVector.h`

- Constructors:
  - `explicit DeviceVector(int size = 0)`
  - `DeviceVector(const DeviceVector& ref)` (deep copy)
  - `DeviceVector(DeviceVector&& other)`
  - `explicit DeviceVector(const HostVector& ref)` (H2D async copy)
- Conversions:
  - `operator HostVector() const` (D2H async copy)
- Methods:
  - `void setConstant(Dtype c)`
  - `void resize(int size)`
  - `void append(const DeviceVector& out)`
  - Equality operators comparing host copies

Context
-------

Header: `src/core/lib/gpu/Context.h`

Selected methods for CKKS ciphertexts (`Ciphertext`, `CtAccurate`) and plaintexts:

- Multiplication and relinearization:
  - `Ciphertext EvalMultAndRelin(const Ciphertext&, const Ciphertext&, const EvaluationKey&, bool verbose=false) const`
  - `CtAccurate EvalMultAndRelin(const CtAccurate&, const CtAccurate&, const EvaluationKey&, bool verbose=false) const`
  - `Ciphertext EvalMultAndRelinNoRescale(...) const`
  - `void EvalMult(const ..., DeviceVector& res0, DeviceVector& res1, DeviceVector& res2) const`
- Squaring:
  - `Ciphertext EvalSquareAndRelin(const Ciphertext&, const EvaluationKey&) const`
  - `CtAccurate EvalSquareAndRelin(const CtAccurate&, const EvaluationKey&) const`
  - `Ciphertext/CtAccurate EvalSquareAndRelinNoRescale(...) const`
- Automorphisms and rotations:
  - `void AutomorphismTransformInPlace(CtAccurate&, const std::vector<uint32_t>& inds) const`
  - `void AutomorphismTransformInPlace(CtAccurate&, uint32_t auto_index) const`
  - `DeviceVector AutomorphismTransform(const DeviceVector&, uint32_t auto_index) const`
- Constant ops and scaling:
  - `Ciphertext/CtAccurate EvalMultConst(...) const`
  - `void EvalMultConstInPlace(...) const`
  - `void EvalAddConstInPlaceWithLoad(CtAccurate&, double, SchemeType, CryptoParamsType) const`
  - `CtAccurate EvalAddConstWithLoad(...) const`
  - `void EvalSubConstInPlaceWithLoad(...) const`
  - `CtAccurate EvalSubConstWithLoad(...) const`
- Add/Sub and key switching:
  - `void Add(const Ciphertext&, const Ciphertext&, Ciphertext&) const`
  - `void Add(const CtAccurate&, const CtAccurate&, CtAccurate&) const`
  - `void Sub(const Ciphertext&, const Ciphertext&, Ciphertext&) const`
  - `void KeySwitch(const DeviceVector& modup_out, const EvaluationKey&, DeviceVector& sum_ax, DeviceVector& sum_bx) const`

MemoryPool
----------

Header: `src/core/lib/gpu/MemoryPool.h`

- `MemoryPool(const Parameter& params)` constructs an RMM binning resource.
- `void UseMemoryPool(bool use)` toggles the custom memory resource for the current device.
- Predefined setups: `defaultMemorySetup`, `mediumMemorySetup`, `bigMemorySetup` configure bins heuristically based on CKKS params (degree, alpha, dnum, special moduli).

Utilities
---------

Header: `src/core/lib/gpu/Utils.h`

- Device transfers:
  - `DeviceVector loadIntoDeviceVector(const std::vector<DCRTPoly>&, bool verbose=false)`
  - `DCRTPoly loadIntoDCRTPoly(const DeviceVector&, std::shared_ptr<M4DCRTParams>, Format=EVALUATION, bool=false)`
- Context generation:
  - `template <typename CryptoParamsType> ckks::Context GenGPUContext(const std::shared_ptr<CryptoParamsType>&)`
- Key/ciphertext loading:
  - `template <typename KeyType> ckks::EvaluationKey LoadRelinKey(const KeyType&)`
  - `template <typename CryptoContextType> ckks::EvaluationKey LoadEvalMultRelinKey(const CryptoContextType&, const std::string keyTag="")`
  - `template <typename CiphertextType> ckks::Ciphertext LoadCiphertext(const CiphertextType&, bool=false)`
  - `template <typename CiphertextType> void LoadCtAccurateFromGPU(CiphertextType&, const ckks::CtAccurate&, std::shared_ptr<M4DCRTParams>)`

Notes
-----

- All host/device transfers are asynchronous on `cudaStreamLegacy`; synchronize as needed in application code.
- Methods marked with templates require appropriate `SchemeType` and `CryptoParamsType` from the CPU-side OpenFHE context.

