//==================================================================================
// LayerNorm approximation under CKKS
//==================================================================================

#include "he_gpt2/layernorm.h"
#include "he_gpt2/packing.h"
#include "he_gpt2/activations.h"

using lbcrypto::Ciphertext;
using lbcrypto::ConstCiphertext;
using lbcrypto::ConstPlaintext;
using lbcrypto::CryptoContext;
using lbcrypto::DCRTPoly;

namespace hegpt2 {

Ciphertext<DCRTPoly> layerNorm(
    const CryptoContext<DCRTPoly>& cc,
    ConstCiphertext<DCRTPoly> x,
    uint32_t width,
    ConstPlaintext gamma,
    ConstPlaintext beta,
    const LayerNormParams& p) {
    // mean
    auto sum = reduceSumSlots(cc, x, width);
    auto mean = cc->EvalMult(sum, 1.0 / static_cast<double>(width));

    // x - mean
    auto x_centered = cc->EvalSub(x, mean);

    // variance ~ mean((x-mean)^2)
    auto sq = cc->EvalMult(x_centered, x_centered);
    auto sumsq = reduceSumSlots(cc, sq, width);
    auto var = cc->EvalMult(sumsq, 1.0 / static_cast<double>(width));

    // inv_sqrt(var + eps) ~ NR on reciprocal sqrt is deeper; use reciprocal on sqrt(var+eps) ~ 1/sqrt via 2 steps
    auto var_eps = cc->EvalAdd(var, p.epsilon);
    // Use reciprocal of sqrt via: y0 = invInit; y = y*(1.5 - 0.5*(var+eps)*y^2)
    auto y = cc->EvalMult(var_eps, 0.0);
    y = cc->EvalAdd(y, p.invInit);
    for (int i = 0; i < 2; ++i) {
        auto y2 = cc->EvalMult(y, y);
        auto t = cc->EvalMult(var_eps, y2);
        auto inner = cc->EvalAdd(1.5, cc->EvalMult(-0.5, t));
        y = cc->EvalMult(y, inner);
    }

    // normalize and affine
    auto x_hat = cc->EvalMult(x_centered, y);
    auto scaled = cc->EvalMult(x_hat, gamma);
    auto shifted = cc->EvalAdd(scaled, beta);
    return shifted;
}

} // namespace hegpt2


