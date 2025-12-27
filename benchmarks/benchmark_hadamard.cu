/**
 * Benchmark to demonstrate kernel launch overhead from cudaDeviceSynchronize()
 * 
 * Key finding: Removing per-gate synchronization gives ~2.3x speedup because
 * CUDA streams automatically serialize operations on the same stream.
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include "StateVector.cuh"
#include "Gates.cuh"
#include "Constants.hpp"

using namespace qsim;

void benchmarkWithSync(int n_qubits, int n_gates) {
    StateVector sv(n_qubits);
    
    cuDoubleComplex* d_state = sv.devicePtr();
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(n_pairs, threads);
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        applyH<<<blocks, threads>>>(d_state, n_qubits, 0);
        cudaDeviceSynchronize();
    }
    
    // Benchmark with sync after each gate
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < n_gates; ++i) {
        int target = i % n_qubits;
        applyH<<<blocks, threads>>>(d_state, n_qubits, target);
        cudaDeviceSynchronize();  // <-- This is the bottleneck!
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << std::setw(15) << "With sync"
              << std::setw(10) << n_qubits
              << std::setw(10) << n_gates
              << std::setw(12) << std::fixed << std::setprecision(2) << ms << " ms"
              << std::endl;
}

void benchmarkAsync(int n_qubits, int n_gates) {
    StateVector sv(n_qubits);
    
    cuDoubleComplex* d_state = sv.devicePtr();
    size_t n_pairs = 1ULL << (n_qubits - 1);
    
    int threads = cuda_config::DEFAULT_BLOCK_SIZE;
    int blocks = calcBlocks(n_pairs, threads);
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        applyH<<<blocks, threads>>>(d_state, n_qubits, 0);
    }
    cudaDeviceSynchronize();
    
    // Benchmark without sync between gates (only at end)
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < n_gates; ++i) {
        int target = i % n_qubits;
        applyH<<<blocks, threads>>>(d_state, n_qubits, target);
        // No sync here - CUDA handles serialization automatically
    }
    cudaDeviceSynchronize();  // Only sync at the end
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << std::setw(15) << "Async (end)"
              << std::setw(10) << n_qubits
              << std::setw(10) << n_gates
              << std::setw(12) << std::fixed << std::setprecision(2) << ms << " ms"
              << std::endl;
}

int main() {
    std::cout << "=== Kernel Launch Overhead Benchmark ===" << std::endl;
    std::cout << "\nThis benchmark demonstrates the performance impact of calling\n"
              << "cudaDeviceSynchronize() after each gate vs. only at the end.\n" << std::endl;
    
    std::cout << std::setw(15) << "Mode"
              << std::setw(10) << "Qubits"
              << std::setw(10) << "Gates"
              << std::setw(12) << "Time"
              << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    for (int n : {16, 20, 22, 24}) {
        benchmarkWithSync(n, 1000);
        benchmarkAsync(n, 1000);
        
        // Report speedup
        StateVector sv1(n), sv2(n);
        cuDoubleComplex* d1 = sv1.devicePtr();
        cuDoubleComplex* d2 = sv2.devicePtr();
        size_t n_pairs = 1ULL << (n - 1);
        int threads = cuda_config::DEFAULT_BLOCK_SIZE;
        int blocks = calcBlocks(n_pairs, threads);
        
        // Time with sync
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; ++i) {
            applyH<<<blocks, threads>>>(d1, n, i % n);
            cudaDeviceSynchronize();
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        double ms_sync = std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        // Time async
        t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; ++i) {
            applyH<<<blocks, threads>>>(d2, n, i % n);
        }
        cudaDeviceSynchronize();
        t2 = std::chrono::high_resolution_clock::now();
        double ms_async = std::chrono::duration<double, std::milli>(t2 - t1).count();
        
        std::cout << std::setw(15) << "  -> Speedup:"
                  << std::setw(10) << ""
                  << std::setw(10) << ""
                  << std::setw(12) << std::fixed << std::setprecision(2) 
                  << (ms_sync / ms_async) << "x"
                  << std::endl;
        std::cout << std::endl;
    }
    
    return 0;
}
