//==================================================================================
// Slot packing helpers and reductions for CKKS
//==================================================================================

#pragma once

#include <cstdint>
#include <vector>

#include "openfhe.h"

namespace hegpt2 {

// Return rotation steps needed for power-of-two reduction over "count" slots
std::vector<int> powerOfTwoRotations(uint32_t count);

// Sum-reduce first "count" slots using power-of-two rotations.
// Assumes rotation keys for steps returned by powerOfTwoRotations(count).
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> reduceSumSlots(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly> x,
    uint32_t count);

// Broadcast the average of first count slots across all used slots.
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> broadcastMean(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly> x,
    uint32_t count);

// Create a plaintext vector of size slots filled with constant c.
lbcrypto::Plaintext makeConstantPlain(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    double c,
    uint32_t slots);

// Compute diagonal offsets for a dense matrix W(outDim x inDim) using the
// diagonal method. Returns vector of offsets in [0, inDim).
std::vector<int> computeDiagOffsets(uint32_t inDim, uint32_t outDim);

} // namespace hegpt2


