//==================================================================================
// GPT-2 model orchestration under CKKS with GPU bootstrapping
//==================================================================================

#pragma once

#include <cstdint>
#include <vector>

#include "openfhe.h"
#include "he_gpt2/ckks_gpu_context.h"
#include "he_gpt2/weights.h"
#include "he_gpt2/attention.h"
#include "he_gpt2/activations.h"
#include "he_gpt2/layernorm.h"

namespace hegpt2 {

struct BootstrapPolicy {
    // Bootstrap every k transformer blocks
    uint32_t everyKBlocks = 2;
    // iterations per bootstrap
    uint32_t numIterations = 1;
};

struct TransformerBlock {
    BlockWeightsDiag W;
};

struct Gpt2Model {
    HeContext he;
    Gpt2Dims dims;
    AttentionConfig attnCfg;
    BootstrapPolicy boot;
    std::vector<TransformerBlock> blocks;
};

// Forward one block on a sequence of ciphertext token vectors X
std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> forwardBlock(
    const Gpt2Model& model,
    const TransformerBlock& block,
    const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& X);

// Full forward pass across all blocks; bootstraps as per policy
std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> forward(
    const Gpt2Model& model,
    const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& X0);

} // namespace hegpt2


