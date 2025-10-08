//==================================================================================
// GPT-2 CKKS GPU context setup (OpenFHE)
//==================================================================================

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openfhe.h"
#include "gpu/Utils.h"

namespace hegpt2 {

struct GpuBootstrapConfig {
    // CoeffsToSlots, SlotsToCoeffs level budgets
    std::vector<uint32_t> levelBudget{4, 4};
    // BSGS dims, 0 means auto
    std::vector<uint32_t> bsgsDim{0, 0};
    // After a bootstrap, how many levels remain available for computation
    uint32_t levelsAfterBootstrap = 45;
    // Number of slots to use (sparse packing). Set to power of two.
    uint32_t numSlots = 1 << 10;
    // Bootstrapping iterations (1 or 2)
    uint32_t numIterations = 1;
};

struct CryptoParamsConfig {
    // NATIVEINT==128 bootstrapping requires FIXEDAUTO or FIXEDMANUAL
    lbcrypto::ScalingTechnique scaling = lbcrypto::FIXEDAUTO;
    // For 128-bit path typical values: 78/89; for 64-bit: 59/60
    uint32_t dcrtBits = 78;
    uint32_t firstMod = 89;
    // Start with strong or NotSet for development
    lbcrypto::SecurityLevel security = lbcrypto::HEStd_128_classic;
    // HYBRID keyswitch with 3 digits works well for bootstrapping
    uint32_t numLargeDigits = 3;
    lbcrypto::KeySwitchTechnique ksTech = lbcrypto::HYBRID;
    // Ring dimension; if 0, OpenFHE will infer from security
    uint32_t ringDim = 0; // e.g., 1<<17 if you want to force N
};

struct HeContext {
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keyPair;
    // GPU context used by EvalBootstrapGPU and potential GPU helpers
    ckks::Context gpu;
    // PublicKey keyTag helps fetch rotation keys
    std::string keyTag;
};

// Build CKKS crypto context with bootstrapping enabled, generate keys, and
// initialize the GPU context with preloaded eval mult/relin and rotation keys.
// The number of rotation keys cached on GPU can be capped by maxRotationKeys
// (negative => load all available).
HeContext buildHeGpuContext(const CryptoParamsConfig& paramsCfg,
                            const GpuBootstrapConfig& bootCfg,
                            int maxRotationKeys = -1);

// Schedule a GPU bootstrapping on the given ciphertext.
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> bootstrapGpu(
    const HeContext& he,
    lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly> ct,
    uint32_t numIterations = 1,
    uint32_t precision = 0);

} // namespace hegpt2


