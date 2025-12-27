#include "Circuit.hpp"
#include "Constants.hpp"
#include <stdexcept>
#include <sstream>
#include <random>
#include <algorithm>
#include <cmath>

namespace qsim {

Circuit::Circuit(int num_qubits) : num_qubits_(num_qubits) {
    if (!isValidQubitCount(num_qubits)) {
        throw std::invalid_argument(
            "Number of qubits must be between " + 
            std::to_string(cuda_config::MIN_QUBITS) + " and " + 
            std::to_string(cuda_config::MAX_QUBITS)
        );
    }
}

void Circuit::validateQubit(int qubit) const {
    if (qubit < 0 || qubit >= num_qubits_) {
        throw std::out_of_range("Qubit index " + std::to_string(qubit) + 
                                 " out of range [0, " + std::to_string(num_qubits_ - 1) + "]");
    }
}

void Circuit::validateQubitPair(int q1, int q2) const {
    validateQubit(q1);
    validateQubit(q2);
    if (q1 == q2) {
        throw std::invalid_argument("Two-qubit gate requires distinct qubits");
    }
}

void Circuit::validateQubitTriple(int q1, int q2, int q3) const {
    validateQubit(q1);
    validateQubit(q2);
    validateQubit(q3);
    if (q1 == q2 || q1 == q3 || q2 == q3) {
        throw std::invalid_argument("Three-qubit gate requires three distinct qubits");
    }
}

namespace {
    void validateAngle(double theta) {
        if (std::isnan(theta) || std::isinf(theta)) {
            throw std::invalid_argument("Rotation angle must be a finite number");
        }
    }
}

Circuit& Circuit::x(int qubit) {
    validateQubit(qubit);
    gates_.emplace_back(GateType::X, qubit);
    return *this;
}

Circuit& Circuit::y(int qubit) {
    validateQubit(qubit);
    gates_.emplace_back(GateType::Y, qubit);
    return *this;
}

Circuit& Circuit::z(int qubit) {
    validateQubit(qubit);
    gates_.emplace_back(GateType::Z, qubit);
    return *this;
}

Circuit& Circuit::h(int qubit) {
    validateQubit(qubit);
    gates_.emplace_back(GateType::H, qubit);
    return *this;
}

Circuit& Circuit::s(int qubit) {
    validateQubit(qubit);
    gates_.emplace_back(GateType::S, qubit);
    return *this;
}

Circuit& Circuit::t(int qubit) {
    validateQubit(qubit);
    gates_.emplace_back(GateType::T, qubit);
    return *this;
}

Circuit& Circuit::sdag(int qubit) {
    validateQubit(qubit);
    gates_.emplace_back(GateType::Sdag, qubit);
    return *this;
}

Circuit& Circuit::tdag(int qubit) {
    validateQubit(qubit);
    gates_.emplace_back(GateType::Tdag, qubit);
    return *this;
}

Circuit& Circuit::rx(int qubit, double theta) {
    validateQubit(qubit);
    validateAngle(theta);
    gates_.emplace_back(GateType::Rx, qubit, theta);
    return *this;
}

Circuit& Circuit::ry(int qubit, double theta) {
    validateQubit(qubit);
    validateAngle(theta);
    gates_.emplace_back(GateType::Ry, qubit, theta);
    return *this;
}

Circuit& Circuit::rz(int qubit, double theta) {
    validateQubit(qubit);
    validateAngle(theta);
    gates_.emplace_back(GateType::Rz, qubit, theta);
    return *this;
}

Circuit& Circuit::cnot(int control, int target) {
    validateQubitPair(control, target);
    gates_.emplace_back(GateType::CNOT, control, target);
    return *this;
}

Circuit& Circuit::cz(int control, int target) {
    validateQubitPair(control, target);
    gates_.emplace_back(GateType::CZ, control, target);
    return *this;
}

Circuit& Circuit::swap(int qubit1, int qubit2) {
    validateQubitPair(qubit1, qubit2);
    gates_.emplace_back(GateType::SWAP, qubit1, qubit2);
    return *this;
}

Circuit& Circuit::cry(int control, int target, double theta) {
    validateQubitPair(control, target);
    validateAngle(theta);
    gates_.emplace_back(GateType::CRY, control, target, theta);
    return *this;
}

Circuit& Circuit::crz(int control, int target, double theta) {
    validateQubitPair(control, target);
    validateAngle(theta);
    gates_.emplace_back(GateType::CRZ, control, target, theta);
    return *this;
}

Circuit& Circuit::toffoli(int control1, int control2, int target) {
    validateQubitTriple(control1, control2, target);
    gates_.emplace_back(GateType::Toffoli, control1, control2, target);
    return *this;
}

size_t Circuit::getDepth() const {
    if (gates_.empty()) return 0;
    
    // Track the "time" at which each qubit becomes free
    std::vector<size_t> qubit_depth(num_qubits_, 0);
    
    for (const auto& gate : gates_) {
        size_t max_depth = 0;
        for (int q : gate.qubits) {
            max_depth = std::max(max_depth, qubit_depth[q]);
        }
        for (int q : gate.qubits) {
            qubit_depth[q] = max_depth + 1;
        }
    }
    
    return *std::max_element(qubit_depth.begin(), qubit_depth.end());
}

std::string Circuit::toString() const {
    std::ostringstream oss;
    oss << "Circuit(" << num_qubits_ << " qubits, " << gates_.size() << " gates):\n";
    
    auto gateName = [](GateType t) -> const char* {
        switch (t) {
            case GateType::X: return "X";
            case GateType::Y: return "Y";
            case GateType::Z: return "Z";
            case GateType::H: return "H";
            case GateType::S: return "S";
            case GateType::T: return "T";
            case GateType::Sdag: return "Sdag";
            case GateType::Tdag: return "Tdag";
            case GateType::Rx: return "Rx";
            case GateType::Ry: return "Ry";
            case GateType::Rz: return "Rz";
            case GateType::CNOT: return "CNOT";
            case GateType::CZ: return "CZ";
            case GateType::CRY: return "CRY";
            case GateType::CRZ: return "CRZ";
            case GateType::SWAP: return "SWAP";
            case GateType::Toffoli: return "Toffoli";
            default: return "?";
        }
    };
    
    for (size_t i = 0; i < gates_.size(); ++i) {
        const auto& g = gates_[i];
        oss << "  " << i << ": " << gateName(g.type) << "(";
        for (size_t j = 0; j < g.qubits.size(); ++j) {
            if (j > 0) oss << ", ";
            oss << g.qubits[j];
        }
        bool has_param = (g.type == GateType::Rx || g.type == GateType::Ry || 
                          g.type == GateType::Rz || g.type == GateType::CRY ||
                          g.type == GateType::CRZ);
        if (has_param) {
            oss << ", " << g.parameter;
        }
        oss << ")\n";
    }
    
    return oss.str();
}

// ============================================================================
// Factory Functions
// ============================================================================

Circuit createBellCircuit() {
    Circuit c(2);
    c.h(0).cnot(0, 1);
    return c;
}

Circuit createGHZCircuit(int num_qubits) {
    if (num_qubits < 2) {
        throw std::invalid_argument("GHZ circuit requires at least 2 qubits");
    }
    Circuit c(num_qubits);
    c.h(0);
    for (int i = 0; i < num_qubits - 1; ++i) {
        c.cnot(i, i + 1);
    }
    return c;
}

Circuit createRandomCircuit(int num_qubits, int depth, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> qubit_dist(0, num_qubits - 1);
    std::uniform_int_distribution<int> gate_dist(0, 3);  // H, X, CNOT, Rz
    std::uniform_real_distribution<double> angle_dist(0.0, constants::TWO_PI);
    
    Circuit c(num_qubits);
    
    for (int d = 0; d < depth; ++d) {
        int gate_type = gate_dist(rng);
        int q1 = qubit_dist(rng);
        
        switch (gate_type) {
            case 0: c.h(q1); break;
            case 1: c.x(q1); break;
            case 2: {
                if (num_qubits > 1) {
                    int q2 = qubit_dist(rng);
                    while (q2 == q1) q2 = qubit_dist(rng);
                    c.cnot(q1, q2);
                } else {
                    c.h(q1);
                }
                break;
            }
            case 3: c.rz(q1, angle_dist(rng)); break;
        }
    }
    
    return c;
}

} // namespace qsim
