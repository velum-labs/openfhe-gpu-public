//==================================================================================
// LayerNorm approximation under CKKS
//==================================================================================

#pragma once

#include <cstdint>

#include "openfhe.h"

namespace hegpt2 {

struct LayerNormParams {
    double epsilon = 1e-5;
    // Initial guess for reciprocal of variance+eps
    double invInit = 1.0;
};

// x_norm = (x - mean(x)) * inv_sqrt(var(x) + eps); then scale/shift by gamma,beta
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> layerNorm(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly> x,
    uint32_t width,
    lbcrypto::ConstPlaintext gamma,
    lbcrypto::ConstPlaintext beta,
    const LayerNormParams& p = {});

} // namespace hegpt2


