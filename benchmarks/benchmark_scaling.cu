/**
 * Scaling Benchmarks
 * 
 * Measures how performance scales with qubit count.
 * Also compares GPU vs CPU simulator.
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include "Simulator.hpp"

using namespace qsim;

void benchmarkScaling() {
    std::cout << "\n=== Qubit Scaling (GPU) ===\n";
    std::cout << std::setw(10) << "Qubits" 
              << std::setw(15) << "States"
              << std::setw(15) << "Memory (MB)"
              << std::setw(15) << "Init (ms)"
              << std::setw(15) << "100 H gates"
              << "\n";
    std::cout << std::string(70, '-') << "\n";
    
    for (int n = 10; n <= 26; n += 2) {
        size_t states = 1ULL << n;
        double memory_mb = states * 16.0 / (1024 * 1024);
        
        // Time initialization
        auto t0 = std::chrono::high_resolution_clock::now();
        Simulator sim(n);
        auto t1 = std::chrono::high_resolution_clock::now();
        double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        // Time 100 Hadamard gates
        Circuit c(n);
        for (int i = 0; i < 100; ++i) {
            c.h(i % n);
        }
        
        auto t2 = std::chrono::high_resolution_clock::now();
        sim.run(c);
        auto t3 = std::chrono::high_resolution_clock::now();
        double gate_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        
        std::cout << std::setw(10) << n
                  << std::setw(15) << states
                  << std::setw(15) << std::fixed << std::setprecision(1) << memory_mb
                  << std::setw(15) << std::setprecision(2) << init_ms
                  << std::setw(15) << gate_ms
                  << "\n";
    }
}

void benchmarkGPUvsCPU() {
    std::cout << "\n=== GPU vs CPU Comparison ===\n";
    std::cout << std::setw(10) << "Qubits"
              << std::setw(15) << "GPU (ms)"
              << std::setw(15) << "CPU (ms)"
              << std::setw(15) << "Speedup"
              << "\n";
    std::cout << std::string(55, '-') << "\n";
    
    for (int n = 10; n <= 22; n += 2) {
        Circuit c(n);
        for (int i = 0; i < 100; ++i) {
            c.h(i % n);
            if (n > 1 && i % 5 == 0) {
                c.cnot(i % n, (i + 1) % n);
            }
        }
        
        // GPU
        Simulator gpu_sim(n);
        auto t0 = std::chrono::high_resolution_clock::now();
        gpu_sim.run(c);
        auto t1 = std::chrono::high_resolution_clock::now();
        double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        // CPU
        CPUSimulator cpu_sim(n);
        auto t2 = std::chrono::high_resolution_clock::now();
        cpu_sim.run(c);
        auto t3 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        
        double speedup = cpu_ms / gpu_ms;
        
        std::cout << std::setw(10) << n
                  << std::setw(15) << std::fixed << std::setprecision(2) << gpu_ms
                  << std::setw(15) << cpu_ms
                  << std::setw(15) << std::setprecision(1) << speedup << "x"
                  << "\n";
    }
}

int main() {
    std::cout << "=== Scaling Benchmarks ===\n";
    
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n\n";
    
    benchmarkScaling();
    benchmarkGPUvsCPU();
    
    std::cout << "\n=== Complete ===\n";
    return 0;
}
