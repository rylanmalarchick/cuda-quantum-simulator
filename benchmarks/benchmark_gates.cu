// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Rylan Malarchick

/**
 * Gate Benchmarks
 * 
 * Measures throughput for different gate types.
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include "Simulator.hpp"

using namespace qsim;

struct BenchmarkResult {
    std::string name;
    int num_qubits;
    int num_gates;
    double time_ms;
    double gates_per_second;
};

void printResult(const BenchmarkResult& r) {
    std::cout << std::setw(20) << r.name 
              << std::setw(10) << r.num_qubits << " qubits"
              << std::setw(12) << r.num_gates << " gates"
              << std::setw(12) << std::fixed << std::setprecision(2) << r.time_ms << " ms"
              << std::setw(15) << std::scientific << std::setprecision(2) 
              << r.gates_per_second << " gates/s"
              << "\n";
}

BenchmarkResult benchmarkGate(const std::string& name, GateType type, 
                               int num_qubits, int num_gates) {
    Simulator sim(num_qubits);
    Circuit c(num_qubits);
    
    // Add gates alternating between qubits
    for (int i = 0; i < num_gates; ++i) {
        int target = i % num_qubits;
        switch (type) {
            case GateType::H: c.h(target); break;
            case GateType::X: c.x(target); break;
            case GateType::Rz: c.rz(target, 0.5); break;
            default: break;
        }
    }
    
    // Warm up
    sim.run(c);
    sim.reset();
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    sim.run(c);
    auto end = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {name, num_qubits, num_gates, ms, num_gates / (ms / 1000.0)};
}

BenchmarkResult benchmarkCNOT(int num_qubits, int num_gates) {
    if (num_qubits < 2) return {"CNOT", num_qubits, 0, 0, 0};
    
    Simulator sim(num_qubits);
    Circuit c(num_qubits);
    
    for (int i = 0; i < num_gates; ++i) {
        int control = i % (num_qubits - 1);
        int target = control + 1;
        c.cnot(control, target);
    }
    
    sim.run(c);
    sim.reset();
    
    auto start = std::chrono::high_resolution_clock::now();
    sim.run(c);
    auto end = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    return {"CNOT", num_qubits, num_gates, ms, num_gates / (ms / 1000.0)};
}

int main() {
    std::cout << "=== Gate Throughput Benchmarks ===\n\n";
    
    std::vector<int> qubit_counts = {10, 15, 20, 24};
    int num_gates = 1000;
    
    for (int n : qubit_counts) {
        std::cout << "\n--- " << n << " Qubits ---\n";
        printResult(benchmarkGate("Hadamard (H)", GateType::H, n, num_gates));
        printResult(benchmarkGate("Pauli-X", GateType::X, n, num_gates));
        printResult(benchmarkGate("Rotation (Rz)", GateType::Rz, n, num_gates));
        printResult(benchmarkCNOT(n, num_gates));
    }
    
    std::cout << "\n=== Benchmark Complete ===\n";
    return 0;
}
