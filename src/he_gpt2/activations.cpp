//==================================================================================
// Polynomial activations and small rational ops under CKKS
//==================================================================================

#include "he_gpt2/activations.h"

using lbcrypto::Ciphertext;
using lbcrypto::ConstCiphertext;
using lbcrypto::CryptoContext;
using lbcrypto::DCRTPoly;

namespace hegpt2 {

Ciphertext<DCRTPoly> evalPoly(
    const CryptoContext<DCRTPoly>& cc,
    ConstCiphertext<DCRTPoly> x,
    const std::vector<double>& c) {
    // Horner scheme: (((c_n x + c_{n-1}) x + ... ) x + c_0)
    Ciphertext<DCRTPoly> acc;
    bool first = true;
    for (int i = static_cast<int>(c.size()) - 1; i >= 0; --i) {
        if (first) {
            acc = cc->EvalMult(x, 0.0); // zero-like at same level
            acc = cc->EvalAdd(acc, c[i]);
            first = false;
        } else {
            acc = cc->EvalMult(acc, x);
            acc = cc->EvalAdd(acc, c[i]);
        }
    }
    return acc;
}

Ciphertext<DCRTPoly> geluCubic(
    const CryptoContext<DCRTPoly>& cc,
    ConstCiphertext<DCRTPoly> x) {
    // y = 0.5 * x * (1 + a x + b x^3)
    constexpr double a = 0.79788456;
    constexpr double b = 0.0356774;
    auto x2 = cc->EvalMult(x, x);
    auto x3 = cc->EvalMult(x2, x);
    auto ax = cc->EvalMult(x, a);
    auto bx3 = cc->EvalMult(x3, b);
    auto inner = cc->EvalAdd(cc->EvalAdd(ax, bx3), 1.0);
    auto y = cc->EvalMult(x, inner);
    y = cc->EvalMult(y, 0.5);
    return y;
}

Ciphertext<DCRTPoly> reciprocalNR(
    const CryptoContext<DCRTPoly>& cc,
    ConstCiphertext<DCRTPoly> z,
    double initGuess) {
    // y_{t+1} = y_t * (2 - z * y_t)
    auto y = cc->EvalMult(z, 0.0); // zero at level
    y = cc->EvalAdd(y, initGuess);
    for (int i = 0; i < 2; ++i) {
        auto zy = cc->EvalMult(z, y);
        auto two_minus_zy = cc->EvalAdd(2.0, cc->EvalNegate(zy));
        y = cc->EvalMult(y, two_minus_zy);
    }
    return y;
}

} // namespace hegpt2


