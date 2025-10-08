//==================================================================================
// Polynomial activations and small rational ops under CKKS
//==================================================================================

#pragma once

#include <vector>

#include "openfhe.h"

namespace hegpt2 {

// gelu(x) ~ 0.5 x (1 + a x + b x^3) with a=0.79788456, b=0.0356774
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> geluCubic(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly> x);

// Approximate reciprocal 1/z with 2-step Newton-Raphson around z0 (scalar init)
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> reciprocalNR(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly> z,
    double initGuess);

// Convenience: Evaluate polynomial with coefficients c0 + c1 x + c2 x^2 + ...
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> evalPoly(
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly> x,
    const std::vector<double>& coeffs);

} // namespace hegpt2


