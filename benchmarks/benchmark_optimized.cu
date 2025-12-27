/**
 * Benchmark: Original vs Optimized Gate Kernels
 * 
 * Compares performance of:
 * - Original Hadamard kernel (applyH)
 * - Shared memory optimized (applyH_shared)
 * - Coalesced access optimized (applyH_coalesced)
 * - Auto-dispatched optimized (applyHadamardOptimized)
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>
#include "Gates.cuh"
#include "OptimizedGates.cuh"
#include "StateVector.cuh"

using namespace qsim;

// Warm up GPU
void warmUp() {
    cudaDeviceSynchronize();
    void* ptr;
    cudaMalloc(&ptr, 1024);
    cudaFree(ptr);
    cudaDeviceSynchronize();
}

// Time a kernel execution (multiple iterations)
template<typename Func>
double timeKernel(Func kernel, int iterations = 100) {
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        kernel();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / iterations;
}

void benchmarkHadamard(int n_qubits) {
    std::cout << "\n=== Hadamard Gate Benchmark (" << n_qubits << " qubits) ===" << std::endl;
    std::cout << std::setw(12) << "Target" 
              << std::setw(15) << "Original(ms)"
              << std::setw(15) << "Coalesced(ms)"
              << std::setw(15) << "Shared(ms)"
              << std::setw(12) << "Speedup"
              << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    StateVector state(n_qubits);
    cuDoubleComplex* d_state = state.devicePtr();
    
    size_t n_pairs = 1ULL << (n_qubits - 1);
    size_t n_states = 1ULL << n_qubits;
    int threads = 256;
    int blocks_pairs = (n_pairs + threads - 1) / threads;
    int blocks_states = (n_states + threads - 1) / threads;
    
    // Test different target qubits
    std::vector<int> targets;
    for (int t = 0; t < std::min(n_qubits, 10); ++t) {
        targets.push_back(t);
    }
    if (n_qubits > 10) {
        targets.push_back(n_qubits / 2);
        targets.push_back(n_qubits - 1);
    }
    
    for (int target : targets) {
        state.initializeZero();  // Reset to |0...0>
        
        // Original kernel
        double orig_time = timeKernel([&]() {
            applyH<<<blocks_pairs, threads>>>(d_state, n_qubits, target);
        });
        
        state.initializeZero();
        
        // Coalesced kernel
        double coal_time = timeKernel([&]() {
            applyH_coalesced<<<blocks_pairs, threads>>>(d_state, n_qubits, target);
        });
        
        state.initializeZero();
        
        // Shared memory kernel (only for small targets)
        double shared_time = 0;
        if (target < 8) {
            size_t tile_size = 2 * threads;
            int tile_blocks = (n_states + tile_size - 1) / tile_size;
            size_t shared_mem = tile_size * sizeof(cuDoubleComplex);
            
            shared_time = timeKernel([&]() {
                applyH_shared<<<tile_blocks, threads, shared_mem>>>(d_state, n_qubits, target);
            });
        }
        
        // Best speedup
        double best_opt = coal_time;
        if (shared_time > 0 && shared_time < coal_time) {
            best_opt = shared_time;
        }
        double speedup = orig_time / best_opt;
        
        std::cout << std::setw(12) << target
                  << std::setw(15) << std::fixed << std::setprecision(4) << orig_time
                  << std::setw(15) << coal_time
                  << std::setw(15) << (shared_time > 0 ? std::to_string(shared_time).substr(0,7) : "N/A")
                  << std::setw(12) << std::setprecision(2) << speedup << "x"
                  << std::endl;
    }
}

void benchmarkCNOT(int n_qubits) {
    std::cout << "\n=== CNOT Gate Benchmark (" << n_qubits << " qubits) ===" << std::endl;
    std::cout << std::setw(20) << "Control,Target"
              << std::setw(15) << "Original(ms)"
              << std::setw(15) << "Optimized(ms)"
              << std::setw(12) << "Speedup"
              << std::endl;
    std::cout << std::string(62, '-') << std::endl;
    
    StateVector state(n_qubits);
    cuDoubleComplex* d_state = state.devicePtr();
    
    size_t n_states = 1ULL << n_qubits;
    int threads = 256;
    int blocks = (n_states + threads - 1) / threads;
    
    // Test different control/target combinations
    std::vector<std::pair<int,int>> pairs = {
        {0, 1}, {1, 0}, {0, n_qubits-1}, {n_qubits-1, 0}
    };
    if (n_qubits > 4) {
        pairs.push_back({n_qubits/2, n_qubits/2 + 1});
    }
    
    for (auto [control, target] : pairs) {
        if (control >= n_qubits || target >= n_qubits || control == target) continue;
        
        state.initializeZero();
        
        double orig_time = timeKernel([&]() {
            applyCNOT<<<blocks, threads>>>(d_state, n_qubits, control, target);
        });
        
        state.initializeZero();
        
        double opt_time = timeKernel([&]() {
            applyCNOT_opt<<<blocks, threads>>>(d_state, n_qubits, control, target);
        });
        
        double speedup = orig_time / opt_time;
        
        std::cout << std::setw(20) << (std::to_string(control) + "," + std::to_string(target))
                  << std::setw(15) << std::fixed << std::setprecision(4) << orig_time
                  << std::setw(15) << opt_time
                  << std::setw(12) << std::setprecision(2) << speedup << "x"
                  << std::endl;
    }
}

void benchmarkScaling() {
    std::cout << "\n=== Hadamard Scaling Benchmark ===" << std::endl;
    std::cout << std::setw(10) << "Qubits"
              << std::setw(15) << "States"
              << std::setw(15) << "Original(ms)"
              << std::setw(15) << "Optimized(ms)"
              << std::setw(12) << "Speedup"
              << std::endl;
    std::cout << std::string(67, '-') << std::endl;
    
    for (int n_qubits = 10; n_qubits <= 26; n_qubits += 2) {
        StateVector state(n_qubits);
        cuDoubleComplex* d_state = state.devicePtr();
        
        size_t n_pairs = 1ULL << (n_qubits - 1);
        int threads = 256;
        int blocks = (n_pairs + threads - 1) / threads;
        
        int target = n_qubits / 2;  // Middle qubit
        
        double orig_time = timeKernel([&]() {
            applyH<<<blocks, threads>>>(d_state, n_qubits, target);
        }, 50);
        
        state.initializeZero();
        
        double opt_time = timeKernel([&]() {
            applyHadamardOptimized(d_state, n_qubits, target);
        }, 50);
        
        double speedup = orig_time / opt_time;
        
        std::cout << std::setw(10) << n_qubits
                  << std::setw(15) << (1ULL << n_qubits)
                  << std::setw(15) << std::fixed << std::setprecision(4) << orig_time
                  << std::setw(15) << opt_time
                  << std::setw(12) << std::setprecision(2) << speedup << "x"
                  << std::endl;
    }
}

int main() {
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Memory Bandwidth: " << prop.memoryBusWidth << " bits" << std::endl;
    
    warmUp();
    
    // Benchmark at different sizes
    benchmarkHadamard(20);
    benchmarkHadamard(24);
    
    benchmarkCNOT(20);
    benchmarkCNOT(24);
    
    benchmarkScaling();
    
    std::cout << "\nBenchmark complete!" << std::endl;
    return 0;
}
