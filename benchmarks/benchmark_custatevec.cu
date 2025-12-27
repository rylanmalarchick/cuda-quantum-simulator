/**
 * @file benchmark_custatevec.cu
 * @brief Performance comparison between our simulator and NVIDIA cuStateVec
 * 
 * This benchmark compares gate application performance between:
 * 1. Our custom CUDA kernels
 * 2. NVIDIA's cuStateVec library (part of cuQuantum SDK)
 * 
 * cuStateVec is NVIDIA's highly optimized state vector simulation library.
 * Comparing against it provides a reference point for professional-grade performance.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <custatevec.h>

#include "StateVector.cuh"
#include "Gates.cuh"

using namespace qsim;

// Hadamard matrix (row-major)
static const cuDoubleComplex H_MATRIX[4] = {
    {M_SQRT1_2, 0.0}, {M_SQRT1_2, 0.0},
    {M_SQRT1_2, 0.0}, {-M_SQRT1_2, 0.0}
};

// X (Pauli-X) matrix
static const cuDoubleComplex X_MATRIX[4] = {
    {0.0, 0.0}, {1.0, 0.0},
    {1.0, 0.0}, {0.0, 0.0}
};

#define CHECK_CUSTATEVEC(call) do { \
    custatevecStatus_t status = call; \
    if (status != CUSTATEVEC_STATUS_SUCCESS) { \
        std::cerr << "cuStateVec error at line " << __LINE__ << ": " << status << std::endl; \
        exit(1); \
    } \
} while(0)

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

/**
 * Benchmark helper: time a lambda in milliseconds
 */
template<typename Func>
double benchmark(Func&& f, int warmup = 3, int iterations = 10) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        f();
        cudaDeviceSynchronize();
    }
    
    // Timed runs
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        f();
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

/**
 * Compare single-qubit gate (Hadamard) performance
 */
void benchmarkHadamard(int n_qubits, int target) {
    const size_t n_states = 1ULL << n_qubits;
    
    // Allocate state vectors
    cuDoubleComplex* d_sv_ours;
    cuDoubleComplex* d_sv_custatevec;
    CHECK_CUDA(cudaMalloc(&d_sv_ours, n_states * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc(&d_sv_custatevec, n_states * sizeof(cuDoubleComplex)));
    
    // Initialize to |0...0>
    std::vector<cuDoubleComplex> h_init(n_states, {0.0, 0.0});
    h_init[0] = {1.0, 0.0};
    CHECK_CUDA(cudaMemcpy(d_sv_ours, h_init.data(), n_states * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sv_custatevec, h_init.data(), n_states * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    
    // Setup cuStateVec
    custatevecHandle_t handle;
    CHECK_CUSTATEVEC(custatevecCreate(&handle));
    
    // Get workspace size for cuStateVec
    size_t workspaceSize = 0;
    CHECK_CUSTATEVEC(custatevecApplyMatrixGetWorkspaceSize(
        handle, CUDA_C_64F, n_qubits, H_MATRIX, CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
        1,   // nTargets
        0,   // nControls
        CUSTATEVEC_COMPUTE_64F, &workspaceSize));
    
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));
    }
    
    // cuStateVec uses little-endian, we use big-endian
    int32_t le_target = n_qubits - 1 - target;
    int32_t targets[] = {le_target};
    
    // Kernel launch params for our implementation
    int threads = 256;
    int blocks = (n_states / 2 + threads - 1) / threads;
    
    // Benchmark our implementation
    double our_time = benchmark([&]() {
        applyH<<<blocks, threads>>>(d_sv_ours, n_qubits, target);
    });
    
    // Benchmark cuStateVec
    double custatevec_time = benchmark([&]() {
        custatevecApplyMatrix(
            handle, d_sv_custatevec, CUDA_C_64F, n_qubits,
            H_MATRIX, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
            targets, 1, nullptr, nullptr, 0,
            CUSTATEVEC_COMPUTE_64F, workspace, workspaceSize);
    });
    
    double ratio = custatevec_time / our_time;
    
    std::cout << std::setw(12) << target 
              << std::setw(16) << std::fixed << std::setprecision(4) << our_time
              << std::setw(16) << custatevec_time
              << std::setw(12) << std::setprecision(2) << ratio << "x"
              << std::endl;
    
    // Cleanup
    if (workspace) cudaFree(workspace);
    custatevecDestroy(handle);
    cudaFree(d_sv_ours);
    cudaFree(d_sv_custatevec);
}

/**
 * Compare two-qubit gate (CNOT) performance
 */
void benchmarkCNOT(int n_qubits, int control, int target) {
    const size_t n_states = 1ULL << n_qubits;
    
    // Allocate state vectors
    cuDoubleComplex* d_sv_ours;
    cuDoubleComplex* d_sv_custatevec;
    CHECK_CUDA(cudaMalloc(&d_sv_ours, n_states * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc(&d_sv_custatevec, n_states * sizeof(cuDoubleComplex)));
    
    // Initialize to |0...0>
    std::vector<cuDoubleComplex> h_init(n_states, {0.0, 0.0});
    h_init[0] = {1.0, 0.0};
    CHECK_CUDA(cudaMemcpy(d_sv_ours, h_init.data(), n_states * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sv_custatevec, h_init.data(), n_states * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    
    // Setup cuStateVec
    custatevecHandle_t handle;
    CHECK_CUSTATEVEC(custatevecCreate(&handle));
    
    // For CNOT: target is acted upon, control is a control qubit
    // cuStateVec uses little-endian, we use big-endian - need to convert
    int32_t le_target = n_qubits - 1 - target;
    int32_t le_control = n_qubits - 1 - control;
    int32_t targets[] = {le_target};
    int32_t controls[] = {le_control};
    int32_t controlBitValues[] = {1};  // Control on |1>
    
    // Get workspace size
    size_t workspaceSize = 0;
    CHECK_CUSTATEVEC(custatevecApplyMatrixGetWorkspaceSize(
        handle, CUDA_C_64F, n_qubits, X_MATRIX, CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
        1,   // nTargets
        1,   // nControls
        CUSTATEVEC_COMPUTE_64F, &workspaceSize));
    
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));
    }
    
    // Kernel launch params
    int threads = 256;
    int blocks = (n_states / 2 + threads - 1) / threads;
    
    // Benchmark our implementation
    double our_time = benchmark([&]() {
        applyCNOT<<<blocks, threads>>>(d_sv_ours, n_qubits, control, target);
    });
    
    // Benchmark cuStateVec (controlled-X = CNOT)
    double custatevec_time = benchmark([&]() {
        custatevecApplyMatrix(
            handle, d_sv_custatevec, CUDA_C_64F, n_qubits,
            X_MATRIX, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
            targets, 1, controls, controlBitValues, 1,
            CUSTATEVEC_COMPUTE_64F, workspace, workspaceSize);
    });
    
    double ratio = custatevec_time / our_time;
    
    std::cout << std::setw(8) << control << "," << std::setw(3) << target
              << std::setw(16) << std::fixed << std::setprecision(4) << our_time
              << std::setw(16) << custatevec_time
              << std::setw(12) << std::setprecision(2) << ratio << "x"
              << std::endl;
    
    // Cleanup
    if (workspace) cudaFree(workspace);
    custatevecDestroy(handle);
    cudaFree(d_sv_ours);
    cudaFree(d_sv_custatevec);
}

/**
 * Benchmark scaling with qubit count
 */
void benchmarkScaling() {
    std::cout << "\n=== Hadamard Scaling: Ours vs cuStateVec ===" << std::endl;
    std::cout << std::setw(10) << "Qubits" 
              << std::setw(14) << "States"
              << std::setw(14) << "Ours(ms)"
              << std::setw(16) << "cuStateVec(ms)"
              << std::setw(12) << "Ratio"
              << std::endl;
    std::cout << std::string(66, '-') << std::endl;
    
    for (int n_qubits : {12, 16, 18, 20, 22, 24, 26}) {
        const size_t n_states = 1ULL << n_qubits;
        int target = 0;
        
        // Allocate
        cuDoubleComplex* d_sv_ours;
        cuDoubleComplex* d_sv_custatevec;
        CHECK_CUDA(cudaMalloc(&d_sv_ours, n_states * sizeof(cuDoubleComplex)));
        CHECK_CUDA(cudaMalloc(&d_sv_custatevec, n_states * sizeof(cuDoubleComplex)));
        
        // Initialize
        CHECK_CUDA(cudaMemset(d_sv_ours, 0, n_states * sizeof(cuDoubleComplex)));
        CHECK_CUDA(cudaMemset(d_sv_custatevec, 0, n_states * sizeof(cuDoubleComplex)));
        cuDoubleComplex one = {1.0, 0.0};
        CHECK_CUDA(cudaMemcpy(d_sv_ours, &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_sv_custatevec, &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
        
        // Setup cuStateVec
        custatevecHandle_t handle;
        CHECK_CUSTATEVEC(custatevecCreate(&handle));
        
        int32_t le_target = n_qubits - 1 - target;
        int32_t targets[] = {le_target};
        
        size_t workspaceSize = 0;
        CHECK_CUSTATEVEC(custatevecApplyMatrixGetWorkspaceSize(
            handle, CUDA_C_64F, n_qubits, H_MATRIX, CUDA_C_64F,
            CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
            1, 0, CUSTATEVEC_COMPUTE_64F, &workspaceSize));
        
        void* workspace = nullptr;
        if (workspaceSize > 0) {
            CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));
        }
        
        int threads = 256;
        int blocks = (n_states / 2 + threads - 1) / threads;
        
        // Benchmark
        double our_time = benchmark([&]() {
            applyH<<<blocks, threads>>>(d_sv_ours, n_qubits, target);
        });
        
        double custatevec_time = benchmark([&]() {
            custatevecApplyMatrix(
                handle, d_sv_custatevec, CUDA_C_64F, n_qubits,
                H_MATRIX, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                targets, 1, nullptr, nullptr, 0,
                CUSTATEVEC_COMPUTE_64F, workspace, workspaceSize);
        });
        
        double ratio = custatevec_time / our_time;
        
        std::cout << std::setw(10) << n_qubits
                  << std::setw(14) << n_states
                  << std::setw(14) << std::fixed << std::setprecision(4) << our_time
                  << std::setw(16) << custatevec_time
                  << std::setw(11) << std::setprecision(2) << ratio << "x"
                  << std::endl;
        
        // Cleanup
        if (workspace) cudaFree(workspace);
        custatevecDestroy(handle);
        cudaFree(d_sv_ours);
        cudaFree(d_sv_custatevec);
    }
}

/**
 * Benchmark circuit execution (sequence of gates)
 */
void benchmarkCircuit(int n_qubits, int depth) {
    std::cout << "\n=== Circuit Benchmark: " << n_qubits << " qubits, depth " << depth << " ===" << std::endl;
    
    const size_t n_states = 1ULL << n_qubits;
    
    // Allocate
    cuDoubleComplex* d_sv_ours;
    cuDoubleComplex* d_sv_custatevec;
    CHECK_CUDA(cudaMalloc(&d_sv_ours, n_states * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc(&d_sv_custatevec, n_states * sizeof(cuDoubleComplex)));
    
    // Initialize
    CHECK_CUDA(cudaMemset(d_sv_ours, 0, n_states * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMemset(d_sv_custatevec, 0, n_states * sizeof(cuDoubleComplex)));
    cuDoubleComplex one = {1.0, 0.0};
    CHECK_CUDA(cudaMemcpy(d_sv_ours, &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sv_custatevec, &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    
    // Setup cuStateVec
    custatevecHandle_t handle;
    CHECK_CUSTATEVEC(custatevecCreate(&handle));
    
    int threads = 256;
    int blocks = (n_states / 2 + threads - 1) / threads;
    
    // Circuit: alternating layers of H on all qubits, then CNOT chain
    auto run_our_circuit = [&]() {
        for (int d = 0; d < depth; d++) {
            // Layer of Hadamards
            for (int q = 0; q < n_qubits; q++) {
                applyH<<<blocks, threads>>>(d_sv_ours, n_qubits, q);
            }
            // CNOT chain
            for (int q = 0; q < n_qubits - 1; q++) {
                applyCNOT<<<blocks, threads>>>(d_sv_ours, n_qubits, q, q + 1);
            }
        }
    };
    
    auto run_custatevec_circuit = [&]() {
        for (int d = 0; d < depth; d++) {
            // Layer of Hadamards
            for (int q = 0; q < n_qubits; q++) {
                int32_t targets[] = {n_qubits - 1 - q};
                custatevecApplyMatrix(
                    handle, d_sv_custatevec, CUDA_C_64F, n_qubits,
                    H_MATRIX, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                    targets, 1, nullptr, nullptr, 0,
                    CUSTATEVEC_COMPUTE_64F, nullptr, 0);
            }
            // CNOT chain
            for (int q = 0; q < n_qubits - 1; q++) {
                int32_t targets[] = {n_qubits - 1 - (q + 1)};
                int32_t controls[] = {n_qubits - 1 - q};
                int32_t controlBitValues[] = {1};
                custatevecApplyMatrix(
                    handle, d_sv_custatevec, CUDA_C_64F, n_qubits,
                    X_MATRIX, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
                    targets, 1, controls, controlBitValues, 1,
                    CUSTATEVEC_COMPUTE_64F, nullptr, 0);
            }
        }
    };
    
    double our_time = benchmark(run_our_circuit, 2, 5);
    double custatevec_time = benchmark(run_custatevec_circuit, 2, 5);
    double ratio = custatevec_time / our_time;
    
    int total_gates = depth * (n_qubits + (n_qubits - 1));
    
    std::cout << "Total gates: " << total_gates << std::endl;
    std::cout << "Our time:        " << std::fixed << std::setprecision(3) << our_time << " ms" << std::endl;
    std::cout << "cuStateVec time: " << custatevec_time << " ms" << std::endl;
    std::cout << "Ratio (cuSV/ours): " << std::setprecision(2) << ratio << "x" << std::endl;
    std::cout << "Our throughput:        " << std::setprecision(0) << (total_gates * 1000.0 / our_time) << " gates/s" << std::endl;
    std::cout << "cuStateVec throughput: " << (total_gates * 1000.0 / custatevec_time) << " gates/s" << std::endl;
    
    custatevecDestroy(handle);
    cudaFree(d_sv_ours);
    cudaFree(d_sv_custatevec);
}

int main() {
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "cuStateVec Version: " << CUSTATEVEC_VER_MAJOR << "." 
              << CUSTATEVEC_VER_MINOR << "." << CUSTATEVEC_VER_PATCH << std::endl;
    
    // Hadamard benchmark for different target qubits
    std::cout << "\n=== Hadamard Gate: Ours vs cuStateVec (20 qubits) ===" << std::endl;
    std::cout << std::setw(12) << "Target" 
              << std::setw(16) << "Ours(ms)"
              << std::setw(16) << "cuStateVec(ms)"
              << std::setw(12) << "Ratio"
              << std::endl;
    std::cout << std::string(56, '-') << std::endl;
    
    for (int target : {0, 5, 10, 15, 19}) {
        benchmarkHadamard(20, target);
    }
    
    std::cout << "\n=== Hadamard Gate: Ours vs cuStateVec (24 qubits) ===" << std::endl;
    std::cout << std::setw(12) << "Target" 
              << std::setw(16) << "Ours(ms)"
              << std::setw(16) << "cuStateVec(ms)"
              << std::setw(12) << "Ratio"
              << std::endl;
    std::cout << std::string(56, '-') << std::endl;
    
    for (int target : {0, 6, 12, 18, 23}) {
        benchmarkHadamard(24, target);
    }
    
    // CNOT benchmark
    std::cout << "\n=== CNOT Gate: Ours vs cuStateVec (20 qubits) ===" << std::endl;
    std::cout << std::setw(12) << "Ctrl,Tgt" 
              << std::setw(16) << "Ours(ms)"
              << std::setw(16) << "cuStateVec(ms)"
              << std::setw(12) << "Ratio"
              << std::endl;
    std::cout << std::string(56, '-') << std::endl;
    
    benchmarkCNOT(20, 0, 1);
    benchmarkCNOT(20, 0, 19);
    benchmarkCNOT(20, 10, 11);
    benchmarkCNOT(20, 19, 0);
    
    // Scaling benchmark
    benchmarkScaling();
    
    // Circuit benchmarks
    benchmarkCircuit(20, 10);
    benchmarkCircuit(24, 10);
    
    std::cout << "\nBenchmark complete!" << std::endl;
    
    return 0;
}
