//==================================================================================
// GPT-2 model orchestration under CKKS with GPU bootstrapping
//==================================================================================

#include "he_gpt2/model.h"

using lbcrypto::Ciphertext;
using lbcrypto::ConstCiphertext;
using lbcrypto::CryptoContext;
using lbcrypto::DCRTPoly;

namespace hegpt2 {

std::vector<Ciphertext<DCRTPoly>> forwardBlock(
    const Gpt2Model& model,
    const TransformerBlock& block,
    const std::vector<Ciphertext<DCRTPoly>>& X) {
    const auto& cc = model.he.cc;

    // LayerNorm 1
    std::vector<Ciphertext<DCRTPoly>> Xln(X.size());
    for (size_t t = 0; t < X.size(); ++t) {
        Xln[t] = layerNorm(cc, X[t], model.dims.dModel, block.W.ln1Gamma, block.W.ln1Beta);
    }

    // Self-attention
    AttentionConfig acfg = model.attnCfg;
    acfg.dModel = model.dims.dModel;
    acfg.nHead = model.attnCfg.nHead;
    acfg.seqLen = static_cast<uint32_t>(X.size());

    AttentionWeightsDiag aw{block.W.attnWQ, block.W.attnWK, block.W.attnWV, block.W.attnWO};
    auto AttnOut = attentionForward(cc, acfg, aw, Xln);

    // Residual add
    std::vector<Ciphertext<DCRTPoly>> X1(X.size());
    for (size_t t = 0; t < X.size(); ++t) {
        X1[t] = cc->EvalAdd(X[t], AttnOut[t]);
    }

    // LayerNorm 2
    std::vector<Ciphertext<DCRTPoly>> X1ln(X.size());
    for (size_t t = 0; t < X.size(); ++t) {
        X1ln[t] = layerNorm(cc, X1[t], model.dims.dModel, block.W.ln2Gamma, block.W.ln2Beta);
    }

    // MLP: W1 -> GELU -> W2
    std::vector<Ciphertext<DCRTPoly>> H(X.size());
    for (size_t t = 0; t < X.size(); ++t) {
        auto z1 = applyDenseDiag(cc, block.W.mlpW1, X1ln[t]);
        auto a1 = geluCubic(cc, z1);
        H[t] = applyDenseDiag(cc, block.W.mlpW2, a1);
    }

    // Residual add
    std::vector<Ciphertext<DCRTPoly>> Y(X.size());
    for (size_t t = 0; t < X.size(); ++t) {
        Y[t] = cc->EvalAdd(X1[t], H[t]);
    }
    return Y;
}

std::vector<Ciphertext<DCRTPoly>> forward(
    const Gpt2Model& model,
    const std::vector<Ciphertext<DCRTPoly>>& X0) {
    std::vector<Ciphertext<DCRTPoly>> X = X0;
    for (size_t b = 0; b < model.blocks.size(); ++b) {
        X = forwardBlock(model, model.blocks[b], X);
        if (model.boot.everyKBlocks && ((b + 1) % model.boot.everyKBlocks == 0)) {
            for (size_t t = 0; t < X.size(); ++t) {
                X[t] = bootstrapGpu(model.he, X[t], model.boot.numIterations, 0);
            }
        }
    }
    return X;
}

} // namespace hegpt2


