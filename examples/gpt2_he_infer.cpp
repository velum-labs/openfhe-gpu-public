//==================================================================================
// Example: Encrypted GPT-2 forward (skeleton) with CKKS GPU bootstrapping
//==================================================================================

#include "openfhe.h"
#include "he_gpt2/ckks_gpu_context.h"
#include "he_gpt2/model.h"
#include "he_gpt2/weights.h"

#include <iostream>

using namespace lbcrypto;
using namespace hegpt2;

int main() {
    // 1) Build HE + GPU contexts
    CryptoParamsConfig paramsCfg;
    paramsCfg.scaling = FIXEDAUTO; // 128-bit path constraint
    paramsCfg.dcrtBits = 78;
    paramsCfg.firstMod = 89;
    paramsCfg.security = HEStd_128_classic;
    // paramsCfg.ringDim = 1 << 17; // optionally force

    GpuBootstrapConfig bootCfg;
    bootCfg.levelBudget = {4, 4};
    bootCfg.bsgsDim = {0, 0};
    bootCfg.levelsAfterBootstrap = 45;
    bootCfg.numSlots = 1 << 10; // tune based on packing
    bootCfg.numIterations = 1;

    HeContext he = buildHeGpuContext(paramsCfg, bootCfg, /*maxRotationKeys*/ 64);

    // 2) Model dims and configuration
    Gpt2Dims dims;
    dims.dModel = 768; dims.nHead = 12; dims.dMlp = 3072; dims.seqLen = 128; dims.slots = bootCfg.numSlots;

    Gpt2Model model;
    model.he = he;
    model.dims = dims;
    model.attnCfg.dModel = dims.dModel;
    model.attnCfg.nHead = dims.nHead;
    model.attnCfg.seqLen = dims.seqLen;
    model.boot.everyKBlocks = 2;
    model.boot.numIterations = 1;

    // 3) Load weights (placeholder paths)
    // In a real pipeline, load from files and precompute diagonals.
    // Here we only demonstrate constructing one block with identity-like placeholders.
    std::vector<std::vector<double>> I(dims.dModel, std::vector<double>(dims.dModel, 0.0));
    for (uint32_t i = 0; i < dims.dModel; ++i) I[i][i] = 1.0;
    std::vector<std::vector<double>> W1(dims.dMlp, std::vector<double>(dims.dModel, 0.0));
    for (uint32_t i = 0; i < dims.dMlp && i < dims.dModel; ++i) W1[i][i] = 1.0;
    std::vector<std::vector<double>> W2(dims.dModel, std::vector<double>(dims.dMlp, 0.0));
    for (uint32_t i = 0; i < dims.dModel && i < dims.dMlp; ++i) W2[i][i] = 1.0;
    std::vector<double> gamma(dims.dModel, 1.0), beta(dims.dModel, 0.0);

    TransformerBlock block;
    block.W = buildBlockWeights(he.cc, dims, I, I, I, I, W1, W2, gamma, beta, gamma, beta);
    model.blocks.push_back(block);

    // 4) Encrypt dummy inputs (token embeddings precomputed client-side in practice)
    std::vector<Ciphertext<DCRTPoly>> X(dims.seqLen);
    std::vector<double> tokenVec(dims.slots, 0.0);
    tokenVec[0] = 1.0; // a trivial one-hot-like placeholder
    auto p = he.cc->MakeCKKSPackedPlaintext(tokenVec, 1, 0, nullptr, dims.slots);
    p->SetLength(dims.slots);
    for (uint32_t t = 0; t < dims.seqLen; ++t) {
        X[t] = he.cc->Encrypt(he.keyPair.publicKey, p);
    }

    // 5) Forward through 1 block and a bootstrap (as per policy)
    auto Y = forward(model, X);

    // 6) Decrypt head token just to sanity check ciphertext pipeline
    Plaintext out;
    he.cc->Decrypt(he.keyPair.secretKey, Y[0], &out);
    out->SetLength(dims.slots);
    std::cout << "Decrypted head token (first few slots): ";
    auto v = out->GetCKKSPackedValue();
    for (size_t i = 0; i < std::min<size_t>(8, v.size()); ++i) std::cout << v[i] << " ";
    std::cout << std::endl;

    return 0;
}


