//==================================================================================
// Dense layer via diagonal method
//==================================================================================

#include "he_gpt2/dense_diag.h"

#include <algorithm>

using lbcrypto::Ciphertext;
using lbcrypto::ConstCiphertext;
using lbcrypto::CryptoContext;
using lbcrypto::DCRTPoly;
using lbcrypto::Plaintext;

namespace hegpt2 {

static Plaintext makePlainDiag(const CryptoContext<DCRTPoly>& cc,
                               const std::vector<double>& diag,
                               uint32_t slots) {
    std::vector<double> v(slots, 0.0);
    // Place diagonal entries into first inDim positions; caller pads appropriately
    for (size_t i = 0; i < diag.size() && i < v.size(); ++i) v[i] = diag[i];
    auto p = cc->MakeCKKSPackedPlaintext(v, 1, 0, nullptr, slots);
    p->SetLength(slots);
    return p;
}

DiagonalCacheEntry precomputeDiagonals(
    const CryptoContext<DCRTPoly>& cc,
    const std::vector<std::vector<double>>& W,
    uint32_t inDim,
    uint32_t outDim,
    uint32_t slots) {
    DiagonalCacheEntry cache;
    cache.inDim = inDim;
    cache.outDim = outDim;
    cache.slots = slots;

    // Offsets 0..inDim-1
    cache.steps.resize(inDim);
    for (uint32_t k = 0; k < inDim; ++k) cache.steps[k] = static_cast<int>(k);

    // Build diagonals d_k of length slots
    cache.diagonals.reserve(inDim);
    // For each k, diagonal has entries W[i, (i-k) mod inDim]
    for (uint32_t k = 0; k < inDim; ++k) {
        std::vector<double> diag(inDim, 0.0);
        for (uint32_t i = 0; i < outDim && i < inDim; ++i) {
            uint32_t j = (i + inDim - k) % inDim;
            if (j < inDim) {
                // Clamp to matrix bounds if rectangular
                if (i < outDim && j < inDim) diag[i] = W[i][j];
            }
        }
        cache.diagonals.emplace_back(makePlainDiag(cc, diag, slots));
    }
    return cache;
}

Ciphertext<DCRTPoly> applyDenseDiag(
    const CryptoContext<DCRTPoly>& cc,
    const DiagonalCacheEntry& cache,
    ConstCiphertext<DCRTPoly> x) {
    // y = sum_k Rot(x, k) .* d_k
    Ciphertext<DCRTPoly> acc;
    bool first = true;
    for (size_t k = 0; k < cache.steps.size(); ++k) {
        auto xr = (cache.steps[k] == 0) ? Ciphertext<DCRTPoly>(x->Clone()) : cc->EvalAtIndex(x, cache.steps[k]);
        auto t = cc->EvalMult(xr, cache.diagonals[k]);
        if (first) {
            acc = t;
            first = false;
        } else {
            acc = cc->EvalAdd(acc, t);
        }
    }
    return acc;
}

} // namespace hegpt2


