     //==================================================================================
// GPT-2 plaintext weights loading and diagonal precomputation
//==================================================================================

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "openfhe.h"
#include "he_gpt2/dense_diag.h"

namespace hegpt2 {

struct Gpt2Dims {
    uint32_t dModel = 768;
    uint32_t nHead = 12;
    uint32_t dMlp = 3072;
    uint32_t seqLen = 128;
    uint32_t slots = 1 << 10;
};

struct BlockWeightsDiag {
    DiagonalCacheEntry attnWQ;
    DiagonalCacheEntry attnWK;
    DiagonalCacheEntry attnWV;
    DiagonalCacheEntry attnWO;
    DiagonalCacheEntry mlpW1;
    DiagonalCacheEntry mlpW2;
    lbcrypto::Plaintext ln1Gamma, ln1Beta;
    lbcrypto::Plaintext ln2Gamma, ln2Beta;
};

// Load CSV matrix row-major (out x in)
std::vector<std::vector<double>> loadCsvMatrix(const std::string& path);

// Build a block's diagonal caches from dense matrices
BlockWeightsDiag buildBlockWeights(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const Gpt2Dims& dims,
    const std::vector<std::vector<double>>& WQ,
    const std::vector<std::vector<double>>& WK,
    const std::vector<std::vector<double>>& WV,
    const std::vector<std::vector<double>>& WO,
    const std::vector<std::vector<double>>& W1,
    const std::vector<std::vector<double>>& W2,
    const std::vector<double>& ln1Gamma,
    const std::vector<double>& ln1Beta,
    const std::vector<double>& ln2Gamma,
    const std::vector<double>& ln2Beta);

} // namespace hegpt2


