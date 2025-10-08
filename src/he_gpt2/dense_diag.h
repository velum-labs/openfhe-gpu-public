//==================================================================================
// Dense layer via diagonal method: Wx = sum_k Rot(x, k) .* d_k
//==================================================================================

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openfhe.h"

namespace hegpt2 {

struct DiagonalCacheEntry {
    // Each diagonal as a plaintext vector to multiply slotwise
    std::vector<lbcrypto::Plaintext> diagonals;
    // Rotation steps aligned to diagonals
    std::vector<int> steps;
    uint32_t inDim = 0;
    uint32_t outDim = 0;
    uint32_t slots = 0;
};

// Precompute diagonals for W(out x in) padded to slots.
DiagonalCacheEntry precomputeDiagonals(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const std::vector<std::vector<double>>& W,
    uint32_t inDim,
    uint32_t outDim,
    uint32_t slots);

// Apply cached diagonals to input ciphertext x, producing output ciphertext y.
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> applyDenseDiag(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const DiagonalCacheEntry& cache,
    lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly> x);

} // namespace hegpt2


