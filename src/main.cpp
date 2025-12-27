#include <iostream>
#include <iomanip>
#include "Simulator.hpp"

using namespace qsim;

void printStateVector(const std::vector<std::complex<double>>& state, int num_qubits) {
    std::cout << "State vector (" << state.size() << " amplitudes):\n";
    for (size_t i = 0; i < state.size() && i < 16; ++i) {
        double real = state[i].real();
        double imag = state[i].imag();
        if (std::abs(real) > 1e-10 || std::abs(imag) > 1e-10) {
            std::cout << "  |";
            for (int b = num_qubits - 1; b >= 0; --b) {
                std::cout << ((i >> b) & 1);
            }
            std::cout << "⟩: " << std::fixed << std::setprecision(6) 
                      << real << " + " << imag << "i\n";
        }
    }
    if (state.size() > 16) {
        std::cout << "  ... (showing first 16 states)\n";
    }
}

void printProbabilities(const std::vector<double>& probs, int num_qubits) {
    std::cout << "Probabilities:\n";
    for (size_t i = 0; i < probs.size() && i < 16; ++i) {
        if (probs[i] > 1e-10) {
            std::cout << "  |";
            for (int b = num_qubits - 1; b >= 0; --b) {
                std::cout << ((i >> b) & 1);
            }
            std::cout << "⟩: " << std::fixed << std::setprecision(6) << probs[i] << "\n";
        }
    }
}

int main() {
    std::cout << "=== CUDA Quantum Simulator ===\n\n";
    
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n\n";
    
    // Demo 1: Bell state
    std::cout << "--- Demo 1: Bell State ---\n";
    {
        Simulator sim(2);
        Circuit circuit = createBellCircuit();
        std::cout << circuit.toString() << "\n";
        
        sim.run(circuit);
        
        auto state = sim.getStateVector();
        printStateVector(state, 2);
        
        auto probs = sim.getProbabilities();
        printProbabilities(probs, 2);
        std::cout << "\n";
    }
    
    // Demo 2: GHZ state
    std::cout << "--- Demo 2: GHZ State (4 qubits) ---\n";
    {
        Circuit circuit = createGHZCircuit(4);
        std::cout << circuit.toString() << "\n";
        
        Simulator sim(4);
        sim.run(circuit);
        
        auto probs = sim.getProbabilities();
        printProbabilities(probs, 4);
        std::cout << "\n";
    }
    
    // Demo 3: Sampling
    std::cout << "--- Demo 3: Sampling Bell State (1000 shots) ---\n";
    {
        Simulator sim(2);
        sim.run(createBellCircuit());
        
        auto samples = sim.sample(1000);
        
        // Count outcomes
        int count_00 = 0, count_11 = 0;
        for (int s : samples) {
            if (s == 0) count_00++;
            else if (s == 3) count_11++;
        }
        std::cout << "  |00⟩: " << count_00 << " times\n";
        std::cout << "  |11⟩: " << count_11 << " times\n";
        std::cout << "  (Expected: ~500 each)\n\n";
    }
    
    std::cout << "=== Simulation Complete ===\n";
    return 0;
}
