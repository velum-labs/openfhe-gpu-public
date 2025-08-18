Performance Notes
=================

Design
------
- Fused kernels: combine iNTT/sub/negate/const-mult, keyswitch inner products
- Batched base conversions to reduce kernel launches
- 8-point-per-thread NTT/iNTT to increase arithmetic intensity

Tuning
------
- Use larger ring dimensions to saturate GPU (e.g., N=2^16 or 2^17). Smaller N underutilizes SMs.
- Preload frequently used rotation keys; minimize host-device transfers
- Set ``WITH_OPENMP=ON`` for CPU-side orchestration, but main work runs on GPU

Metrics to track
----------------
- Throughput of EvalMult/EvalSquare and KeySwitch (ops/s)
- Bootstrapping latency (ms) and its subphases
- GPU memory footprint: ciphertext limbs, key cache size

Known bottlenecks
-----------------
- PCIe transfers for large key maps if not cached
- Limited shared memory per SM; kernel configurations balance occupancy and per-thread storage

