//==================================================================================
// GPT-2 plaintext weights loading and diagonal precomputation
//==================================================================================

#include "he_gpt2/weights.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

using lbcrypto::CryptoContext;
using lbcrypto::DCRTPoly;
using lbcrypto::Plaintext;

namespace hegpt2 {

std::vector<std::vector<double>> loadCsvMatrix(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open " + path);
    std::vector<std::vector<double>> M;
    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            row.push_back(std::stod(tok));
        }
        if (!row.empty()) M.push_back(std::move(row));
    }
    return M;
}

static Plaintext plainFromVector(const CryptoContext<DCRTPoly>& cc,
                                 const std::vector<double>& v,
                                 uint32_t slots) {
    std::vector<double> pad(slots, 0.0);
    for (size_t i = 0; i < v.size() && i < pad.size(); ++i) pad[i] = v[i];
    auto p = cc->MakeCKKSPackedPlaintext(pad, 1, 0, nullptr, slots);
    p->SetLength(slots);
    return p;
}

BlockWeightsDiag buildBlockWeights(
    const CryptoContext<DCRTPoly>& cc,
    const Gpt2Dims& dims,
    const std::vector<std::vector<double>>& WQ,
    const std::vector<std::vector<double>>& WK,
    const std::vector<std::vector<double>>& WV,
    const std::vector<std::vector<double>>& WO,
    const std::vector<std::vector<double>>& W1,
    const std::vector<std::vector<double>>& W2,
    const std::vector<double>& ln1Gamma,
    const std::vector<double>& ln1Beta,
    const std::vector<double>& ln2Gamma,
    const std::vector<double>& ln2Beta) {
    BlockWeightsDiag bw;
    bw.attnWQ = precomputeDiagonals(cc, WQ, dims.dModel, dims.dModel, dims.slots);
    bw.attnWK = precomputeDiagonals(cc, WK, dims.dModel, dims.dModel, dims.slots);
    bw.attnWV = precomputeDiagonals(cc, WV, dims.dModel, dims.dModel, dims.slots);
    bw.attnWO = precomputeDiagonals(cc, WO, dims.dModel, dims.dModel, dims.slots);
    bw.mlpW1  = precomputeDiagonals(cc, W1, dims.dModel, dims.dMlp,   dims.slots);
    bw.mlpW2  = precomputeDiagonals(cc, W2, dims.dMlp,   dims.dModel, dims.slots);

    bw.ln1Gamma = plainFromVector(cc, ln1Gamma, dims.slots);
    bw.ln1Beta  = plainFromVector(cc, ln1Beta,  dims.slots);
    bw.ln2Gamma = plainFromVector(cc, ln2Gamma, dims.slots);
    bw.ln2Beta  = plainFromVector(cc, ln2Beta,  dims.slots);
    return bw;
}

} // namespace hegpt2


