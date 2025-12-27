// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Rylan Malarchick

/**
 * @file CudaMemory.cuh
 * @brief RAII wrapper for CUDA device memory management
 * @author Rylan Malarchick
 * @date 2024
 *
 * Provides automatic memory management for GPU allocations following the
 * Resource Acquisition Is Initialization (RAII) idiom. Memory is allocated
 * on construction and automatically freed on destruction, ensuring exception
 * safety and preventing memory leaks.
 *
 * Key features:
 * - Move semantics (no copying) to prevent double-free
 * - Automatic cleanup even when exceptions are thrown
 * - Convenient host<->device transfer methods
 * - Zero-overhead abstraction (raw pointer access for kernels)
 *
 * This wrapper is used by StateVector and DensityMatrix to manage their
 * GPU memory allocations.
 *
 * @note Destructor is noexcept and logs errors rather than throwing to
 *       ensure safe cleanup during stack unwinding.
 */
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace qsim {

/**
 * CudaMemory<T> - RAII wrapper for CUDA device memory
 * 
 * Provides automatic memory management for GPU allocations.
 * Memory is allocated on construction and freed on destruction.
 * Move-only semantics (no copying).
 * 
 * Usage:
 *   CudaMemory<double> d_data(1024);  // Allocate 1024 doubles
 *   double* ptr = d_data.get();       // Get raw pointer for kernel
 *   d_data.copyFromHost(h_data.data(), 1024);
 *   d_data.copyToHost(h_data.data(), 1024);
 */
template <typename T>
class CudaMemory {
public:
    /**
     * Allocate GPU memory for 'count' elements of type T
     * @param count Number of elements to allocate
     * @throws std::runtime_error if allocation fails
     */
    explicit CudaMemory(size_t count)
        : count_(count)
        , ptr_(nullptr)
    {
        if (count_ == 0) {
            return;  // No allocation for zero elements
        }
        
        cudaError_t err = cudaMalloc(&ptr_, count_ * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA malloc failed: ") + cudaGetErrorString(err)
            );
        }
    }
    
    /**
     * Default constructor - creates empty wrapper
     */
    CudaMemory() : count_(0), ptr_(nullptr) {}
    
    /**
     * Destructor - frees GPU memory
     */
    ~CudaMemory() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }
    
    // Disable copy (GPU memory ownership must be unique)
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;
    
    // Enable move
    CudaMemory(CudaMemory&& other) noexcept
        : count_(other.count_)
        , ptr_(other.ptr_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    
    CudaMemory& operator=(CudaMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);
            }
            count_ = other.count_;
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    
    /**
     * Get raw device pointer (for kernel launches)
     */
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    
    /**
     * Get element count
     */
    size_t size() const { return count_; }
    
    /**
     * Get byte count
     */
    size_t bytes() const { return count_ * sizeof(T); }
    
    /**
     * Check if memory is allocated
     */
    bool empty() const { return ptr_ == nullptr; }
    
    /**
     * Copy data from host to device
     * @param src Host pointer to copy from
     * @param count Number of elements to copy (must be <= size())
     * @throws std::runtime_error if copy fails
     */
    void copyFromHost(const T* src, size_t count) {
        if (count > count_) {
            throw std::invalid_argument("Copy count exceeds allocation size");
        }
        if (count == 0) return;
        
        cudaError_t err = cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA memcpy H2D failed: ") + cudaGetErrorString(err)
            );
        }
    }
    
    /**
     * Copy data from device to host
     * @param dst Host pointer to copy to
     * @param count Number of elements to copy (must be <= size())
     * @throws std::runtime_error if copy fails
     */
    void copyToHost(T* dst, size_t count) const {
        if (count > count_) {
            throw std::invalid_argument("Copy count exceeds allocation size");
        }
        if (count == 0) return;
        
        cudaError_t err = cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA memcpy D2H failed: ") + cudaGetErrorString(err)
            );
        }
    }
    
    /**
     * Copy all data to host vector
     * @param dst Vector to copy to (will be resized)
     */
    void copyToHost(std::vector<T>& dst) const {
        dst.resize(count_);
        if (count_ > 0) {
            copyToHost(dst.data(), count_);
        }
    }
    
    /**
     * Set all bytes to zero
     * @throws std::runtime_error if memset fails
     */
    void zero() {
        if (ptr_ && count_ > 0) {
            cudaError_t err = cudaMemset(ptr_, 0, count_ * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("CUDA memset failed: ") + cudaGetErrorString(err)
                );
            }
        }
    }

private:
    size_t count_;
    T* ptr_;
};

} // namespace qsim
