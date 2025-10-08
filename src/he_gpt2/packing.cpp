//==================================================================================
// Slot packing helpers and reductions for CKKS
//==================================================================================

#include "he_gpt2/packing.h"

#include <algorithm>
#include <cmath>

using lbcrypto::Ciphertext;
using lbcrypto::ConstCiphertext;
using lbcrypto::CryptoContext;
using lbcrypto::DCRTPoly;
using lbcrypto::Plaintext;

namespace hegpt2 {

std::vector<int> powerOfTwoRotations(uint32_t count) {
    std::vector<int> steps;
    uint32_t k = 1;
    while (k < count) {
        steps.push_back(static_cast<int>(k));
        k <<= 1U;
    }
    return steps;
}

Ciphertext<DCRTPoly> reduceSumSlots(
    const CryptoContext<DCRTPoly>& cc,
    ConstCiphertext<DCRTPoly> x,
    uint32_t count) {
    auto steps = powerOfTwoRotations(count);
    auto acc = Ciphertext<DCRTPoly>(x->Clone());
    for (int s : steps) {
        auto r = cc->EvalAtIndex(x, s);
        acc = cc->EvalAdd(acc, r);
        x = acc;
    }
    return acc;
}

Ciphertext<DCRTPoly> broadcastMean(
    const CryptoContext<DCRTPoly>& cc,
    ConstCiphertext<DCRTPoly> x,
    uint32_t count) {
    auto sum = reduceSumSlots(cc, x, count);
    double inv = 1.0 / static_cast<double>(count);
    auto mean = cc->EvalMult(sum, inv);
    return mean;
}

Plaintext makeConstantPlain(
    const CryptoContext<DCRTPoly>& cc,
    double c,
    uint32_t slots) {
    std::vector<double> v(slots, c);
    auto p = cc->MakeCKKSPackedPlaintext(v, 1, 0, nullptr, slots);
    p->SetLength(slots);
    return p;
}

std::vector<int> computeDiagOffsets(uint32_t inDim, uint32_t outDim) {
    // Standard diagonal method uses offsets 0..inDim-1
    (void)outDim;
    std::vector<int> offs(inDim);
    for (uint32_t i = 0; i < inDim; ++i) offs[i] = static_cast<int>(i);
    return offs;
}

} // namespace hegpt2


