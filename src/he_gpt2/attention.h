//==================================================================================
// Self-attention with polynomial softmax approximation under CKKS
//==================================================================================

#pragma once

#include <cstdint>
#include <vector>

#include "openfhe.h"
#include "he_gpt2/dense_diag.h"

namespace hegpt2 {

struct AttentionConfig {
    uint32_t dModel = 768;
    uint32_t nHead = 12;
    uint32_t seqLen = 128;
    // cubic exp proxy and row normalization with reciprocal
    std::vector<double> expPoly{1.0, 1.0, 0.5, 1.0 / 6.0};
    double invInit = 0.125; // initial 1/sum guess
};

struct AttentionWeightsDiag {
    // Q, K, V, and output projection W_O
    DiagonalCacheEntry WQ;
    DiagonalCacheEntry WK;
    DiagonalCacheEntry WV;
    DiagonalCacheEntry WO;
};

// Compute multi-head attention for one layer given X (seqLen ciphertexts, each a token vector packed in slots)
// Returns sequence of ciphertexts after attention output projection
std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> attentionForward(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const AttentionConfig& cfg,
    const AttentionWeightsDiag& w,
    const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& X);

} // namespace hegpt2


