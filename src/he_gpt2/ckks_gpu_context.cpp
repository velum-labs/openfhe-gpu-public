//==================================================================================
// GPT-2 CKKS GPU context setup (OpenFHE)
//==================================================================================

#include "he_gpt2/ckks_gpu_context.h"

#include <stdexcept>
#include <utility>

using lbcrypto::CCParams;
using lbcrypto::Ciphertext;
using lbcrypto::ConstCiphertext;
using lbcrypto::CryptoContext;
using lbcrypto::CryptoContextCKKSRNS;
using lbcrypto::CryptoParametersCKKSRNS;
using lbcrypto::DCRTPoly;
using lbcrypto::EvalKey;
using lbcrypto::FHECKKSRNS;
using lbcrypto::HEStd_128_classic;
using lbcrypto::KeyPair;
using lbcrypto::PKE;
using lbcrypto::KEYSWITCH;
using lbcrypto::LEVELEDSHE;
using lbcrypto::ADVANCEDSHE;
using lbcrypto::FHE;
using lbcrypto::Plaintext;
using lbcrypto::SCHEME;
using lbcrypto::SecretKeyDist;
using lbcrypto::UNIFORM_TERNARY;

namespace hegpt2 {

static void enableFeatures(const CryptoContext<DCRTPoly>& cc) {
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);
}

HeContext buildHeGpuContext(const CryptoParamsConfig& paramsCfg,
                            const GpuBootstrapConfig& bootCfg,
                            int maxRotationKeys) {
    // 1) Crypto params
    CCParams<CryptoContextCKKSRNS> parameters;
    // Secret key distribution recommended for CKKS
    SecretKeyDist skDist = UNIFORM_TERNARY;
    parameters.SetSecretKeyDist(skDist);

    parameters.SetSecurityLevel(paramsCfg.security);
    if (paramsCfg.ringDim) {
        parameters.SetRingDim(paramsCfg.ringDim);
    }

    parameters.SetKeySwitchTechnique(paramsCfg.ksTech);
    parameters.SetNumLargeDigits(paramsCfg.numLargeDigits);
    parameters.SetScalingTechnique(paramsCfg.scaling);
    parameters.SetScalingModSize(paramsCfg.dcrtBits);
    parameters.SetFirstModSize(paramsCfg.firstMod);

    // Depth: available after bootstrap + depth to perform a bootstrap
    const uint32_t bootDepth = FHECKKSRNS::GetBootstrapDepth(bootCfg.levelBudget, skDist);
    const uint32_t depth = bootCfg.levelsAfterBootstrap + bootDepth + (bootCfg.numIterations > 1 ? (bootCfg.numIterations - 1) : 0);
    parameters.SetMultiplicativeDepth(depth);

    // 2) Create context, enable features
    HeContext he;
    he.cc = lbcrypto::GenCryptoContext(parameters);
    enableFeatures(he.cc);

    // 3) Bootstrap setup
    he.cc->EvalBootstrapSetup(bootCfg.levelBudget, bootCfg.bsgsDim, bootCfg.numSlots);

    // 4) KeyGen and bootstrap keys
    he.keyPair = he.cc->KeyGen();
    he.keyTag = he.keyPair.publicKey->GetKeyTag();
    he.cc->EvalMultKeyGen(he.keyPair.secretKey);
    he.cc->EvalBootstrapKeyGen(he.keyPair.secretKey, bootCfg.numSlots);

    // 5) GPU context + preloaded keys (Relin + optional subset of rotation keys)
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(he.cc->GetCryptoParameters());
    he.gpu = GenGPUContext(cryptoParams);
    he.gpu.EnableMemoryPool();

    // Load evaluation mult/relin key to GPU
    auto evk = LoadEvalMultRelinKey(he.cc);
    he.gpu.preloaded_evaluation_key = new ckks::EvaluationKey(std::move(evk));

    // Load rotation keys map up to a cap
    const std::map<usint, EvalKey<DCRTPoly>> rotMap = he.cc->GetEvalAutomorphismKeyMap(he.keyTag);
    auto* loaded_rot = new std::map<uint32_t, ckks::EvaluationKey>();
    for (const auto& kv : rotMap) {
        if (maxRotationKeys >= 0 && static_cast<int>(loaded_rot->size()) >= maxRotationKeys)
            break;
        (*loaded_rot)[kv.first] = LoadRelinKey(kv.second);
    }
    he.gpu.preloaded_rotation_key_map = loaded_rot;

    return he;
}

Ciphertext<DCRTPoly> bootstrapGpu(
    const HeContext& he,
    ConstCiphertext<DCRTPoly> ct,
    uint32_t numIterations,
    uint32_t precision) {
    return he.cc->EvalBootstrapGPU(ct, he.gpu, numIterations, precision);
}

} // namespace hegpt2


