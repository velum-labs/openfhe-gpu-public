//==================================================================================
// Self-attention with polynomial softmax approximation under CKKS
//==================================================================================

#include "he_gpt2/attention.h"
#include "he_gpt2/activations.h"
#include "he_gpt2/packing.h"

#include <cmath>

using lbcrypto::Ciphertext;
using lbcrypto::ConstCiphertext;
using lbcrypto::CryptoContext;
using lbcrypto::DCRTPoly;

namespace hegpt2 {

static Ciphertext<DCRTPoly> polyExp(const CryptoContext<DCRTPoly>& cc,
                                    ConstCiphertext<DCRTPoly> x,
                                    const std::vector<double>& c) {
    // exp proxy: 1 + x + 0.5x^2 + (1/6)x^3 by default
    return evalPoly(cc, x, c);
}

std::vector<Ciphertext<DCRTPoly>> attentionForward(
    const CryptoContext<DCRTPoly>& cc,
    const AttentionConfig& cfg,
    const AttentionWeightsDiag& w,
    const std::vector<Ciphertext<DCRTPoly>>& X) {
    const uint32_t L = cfg.seqLen;
    const double scale = 1.0 / std::sqrt(static_cast<double>(cfg.dModel / cfg.nHead));

    // Step 1: Q, K, V projections per token
    std::vector<Ciphertext<DCRTPoly>> Q(L), K(L), V(L);
    for (uint32_t t = 0; t < L; ++t) {
        Q[t] = applyDenseDiag(cc, w.WQ, X[t]);
        K[t] = applyDenseDiag(cc, w.WK, X[t]);
        V[t] = applyDenseDiag(cc, w.WV, X[t]);
    }

    // Step 2: Scores S_{i,j} = <Q_i, K_j>/sqrt(d_k). We fold the scaling into a constant later.
    // We compute per i the row softmax over j, then context c_i = Σ_j softmax(S_i,j) V_j
    std::vector<Ciphertext<DCRTPoly>> Y(L);
    for (uint32_t i = 0; i < L; ++i) {
        // Compute row scores and softmax approximation
        Ciphertext<DCRTPoly> rowSum; bool first = true;
        std::vector<Ciphertext<DCRTPoly>> weights(L);
        for (uint32_t j = 0; j < L; ++j) {
            // Dot product proxy: elementwise product and reduction across feature slots
            auto qk = cc->EvalMult(Q[i], K[j]);
            auto s_ij = reduceSumSlots(cc, qk, w.WQ.slots); // reduce across model dims
            s_ij = cc->EvalMult(s_ij, scale);
            auto w_ij = polyExp(cc, s_ij, cfg.expPoly);
            weights[j] = w_ij;
            if (first) { rowSum = w_ij; first = false; }
            else { rowSum = cc->EvalAdd(rowSum, w_ij); }
        }
        // Normalize
        auto inv = reciprocalNR(cc, rowSum, cfg.invInit);
        // Aggregate context vector
        Ciphertext<DCRTPoly> ctx; bool f2 = true;
        for (uint32_t j = 0; j < L; ++j) {
            auto wj = cc->EvalMult(weights[j], inv);
            auto contrib = cc->EvalMult(wj, V[j]);
            if (f2) { ctx = contrib; f2 = false; }
            else { ctx = cc->EvalAdd(ctx, contrib); }
        }
        // Output projection
        Y[i] = applyDenseDiag(cc, w.WO, ctx);
    }

    return Y;
}

} // namespace hegpt2


