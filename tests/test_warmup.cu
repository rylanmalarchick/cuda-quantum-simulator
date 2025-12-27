/**
 * CUDA Warm-up Exercises
 * 
 * Purpose: Verify CUDA environment works and understand basic concepts before
 * diving into quantum simulation.
 * 
 * Exercises:
 * 1. Vector addition - basic kernel launch, memory transfers
 * 2. Shared memory reduction - understand memory hierarchy
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <numeric>

// ============================================================================
// CUDA Error Checking Helper
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err); \
    } while(0)

// ============================================================================
// Exercise 1: Vector Addition
// ============================================================================

/**
 * Kernel: vectorAdd
 * 
 * Each thread computes one element: c[i] = a[i] + b[i]
 * 
 * Key concepts:
 * - blockIdx.x, blockDim.x, threadIdx.x determine thread ID
 * - We launch more threads than needed, so check bounds
 */
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check - we may have launched extra threads
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

TEST(WarmupTest, VectorAddition) {
    const int N = 1024 * 1024;  // 1M elements
    const size_t bytes = N * sizeof(float);
    
    // Host memory
    std::vector<float> h_a(N), h_b(N), h_c(N);
    
    // Initialize with simple pattern
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }
    
    // Device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    // Copy host -> device
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel
    // Choose 256 threads per block (common choice)
    // Calculate number of blocks needed to cover N elements
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy device -> host
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    
    // Verify results
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        ASSERT_FLOAT_EQ(h_c[i], expected) << "Mismatch at index " << i;
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// ============================================================================
// Exercise 2: Parallel Reduction with Shared Memory
// ============================================================================

/**
 * Kernel: sumReduction
 * 
 * Sum all elements using parallel reduction.
 * Each block reduces its portion to a single value using shared memory.
 * 
 * Key concepts:
 * - Shared memory is fast but limited (~48KB per SM)
 * - __syncthreads() ensures all threads reach same point
 * - Reduction pattern: halve active threads each step
 * 
 * Pattern (for 8 threads):
 * Step 0: [0+4, 1+5, 2+6, 3+7, -, -, -, -]  (threads 0-3 active)
 * Step 1: [0+2, 1+3, -, -, -, -, -, -]      (threads 0-1 active)
 * Step 2: [0+1, -, -, -, -, -, -, -]        (thread 0 active)
 */
__global__ void sumReduction(const float* input, float* blockSums, int n) {
    // Shared memory - declared with 'extern' means size set at launch
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory (0 if out of bounds)
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes block result
    if (tid == 0) {
        blockSums[blockIdx.x] = sdata[0];
    }
}

TEST(WarmupTest, SharedMemoryReduction) {
    const int N = 1024 * 1024;
    const size_t bytes = N * sizeof(float);
    
    // Host data
    std::vector<float> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;  // Sum should be N
    }
    
    // Device data
    float *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
    
    // First reduction: N elements -> numBlocks partial sums
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    float *d_blockSums;
    CUDA_CHECK(cudaMalloc(&d_blockSums, numBlocks * sizeof(float)));
    
    // Launch with shared memory size = threadsPerBlock * sizeof(float)
    size_t sharedMemBytes = threadsPerBlock * sizeof(float);
    sumReduction<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(
        d_input, d_blockSums, N
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Copy partial sums back to host
    std::vector<float> h_blockSums(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_blockSums.data(), d_blockSums, 
                          numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Final reduction on CPU (could also do another kernel)
    float total = std::accumulate(h_blockSums.begin(), h_blockSums.end(), 0.0f);
    
    // Verify
    ASSERT_NEAR(total, static_cast<float>(N), 0.1f) 
        << "Expected sum of " << N << " ones to equal " << N;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_blockSums);
}

// ============================================================================
// Exercise 3: GPU Properties Query
// ============================================================================

TEST(WarmupTest, GPUProperties) {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    ASSERT_GT(deviceCount, 0) << "No CUDA devices found!";
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    std::cout << "\n=== GPU Properties ===\n";
    std::cout << "Name: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Total Global Memory: " << (prop.totalGlobalMem / (1024*1024)) << " MB\n";
    std::cout << "Shared Memory per Block: " << (prop.sharedMemPerBlock / 1024) << " KB\n";
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Warp Size: " << prop.warpSize << "\n";
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << "\n";
    std::cout << "Memory Clock Rate: " << (prop.memoryClockRate / 1000) << " MHz\n";
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
    std::cout << "======================\n\n";
    
    // Basic sanity checks for RTX 4070
    EXPECT_GE(prop.major, 8) << "Expected Ada Lovelace (8.x) or newer";
    EXPECT_GE(prop.totalGlobalMem, 7ULL * 1024 * 1024 * 1024) << "Expected at least 7GB VRAM";
}

// ============================================================================
// Exercise 4: Memory Bandwidth Test
// ============================================================================

__global__ void copyKernel(const float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

TEST(WarmupTest, MemoryBandwidth) {
    const int N = 64 * 1024 * 1024;  // 64M floats = 256MB
    const size_t bytes = N * sizeof(float);
    
    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));
    
    // Warm up
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    copyKernel<<<blocks, threads>>>(d_src, d_dst, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Time multiple copies
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int numIterations = 10;
    
    cudaEventRecord(start);
    for (int i = 0; i < numIterations; ++i) {
        copyKernel<<<blocks, threads>>>(d_src, d_dst, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate bandwidth (read + write = 2 * bytes)
    double seconds = milliseconds / 1000.0;
    double totalBytes = 2.0 * bytes * numIterations;
    double bandwidth = totalBytes / seconds / (1024.0 * 1024.0 * 1024.0);
    
    std::cout << "\n=== Memory Bandwidth ===\n";
    std::cout << "Data size: " << (bytes / (1024*1024)) << " MB\n";
    std::cout << "Time for " << numIterations << " copies: " << milliseconds << " ms\n";
    std::cout << "Effective bandwidth: " << bandwidth << " GB/s\n";
    std::cout << "========================\n\n";
    
    // RTX 4070 Laptop should achieve ~300+ GB/s
    EXPECT_GT(bandwidth, 100.0) << "Bandwidth seems too low";
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);
}
