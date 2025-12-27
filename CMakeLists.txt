cmake_minimum_required(VERSION 3.18)

# Set CUDA compiler before project() call
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(cuda-quantum-simulator LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture - RTX 4070 is Ada Lovelace (sm_89)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Enable position independent code (needed for shared libs)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Debug/Release specific flags
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")  # -G enables device debugging
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ============================================================================
# Main library
# ============================================================================
add_library(quantum_sim_lib STATIC
    src/StateVector.cu
    src/Gates.cu
    src/Circuit.cpp
    src/Simulator.cu
    src/NoiseModel.cu
    src/DensityMatrix.cu
    src/OptimizedGates.cu
)

target_link_libraries(quantum_sim_lib
    cudart
    curand
)

# ============================================================================
# Main executable
# ============================================================================
add_executable(quantum_sim src/main.cpp)
target_link_libraries(quantum_sim quantum_sim_lib)

# ============================================================================
# Google Test
# ============================================================================
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# ============================================================================
# Tests
# ============================================================================

# Warm-up test (vector add, shared memory)
add_executable(test_warmup tests/test_warmup.cu)
target_link_libraries(test_warmup GTest::gtest_main cudart)
add_test(NAME WarmupTest COMMAND test_warmup)

# State vector tests (will add more as we build)
add_executable(test_statevector tests/test_statevector.cu)
target_link_libraries(test_statevector quantum_sim_lib GTest::gtest_main)
add_test(NAME StateVectorTest COMMAND test_statevector)

# Gate tests
add_executable(test_gates tests/test_gates.cu)
target_link_libraries(test_gates quantum_sim_lib GTest::gtest_main)
add_test(NAME GatesTest COMMAND test_gates)

# Gate algebra tests (mathematical identities: H²=I, S²=Z, T⁸=I, etc.)
add_executable(test_gate_algebra tests/test_gate_algebra.cu)
target_link_libraries(test_gate_algebra quantum_sim_lib GTest::gtest_main)
add_test(NAME GateAlgebraTest COMMAND test_gate_algebra)

# GPU vs CPU equivalence tests
add_executable(test_gpu_cpu_equivalence tests/test_gpu_cpu_equivalence.cu)
target_link_libraries(test_gpu_cpu_equivalence quantum_sim_lib GTest::gtest_main)
add_test(NAME GPUCPUEquivalenceTest COMMAND test_gpu_cpu_equivalence)

# Boundary condition tests (edge cases, invalid inputs)
add_executable(test_boundary tests/test_boundary.cu)
target_link_libraries(test_boundary quantum_sim_lib GTest::gtest_main)
add_test(NAME BoundaryTest COMMAND test_boundary)

# Noise model tests
add_executable(test_noise tests/test_noise.cu)
target_link_libraries(test_noise quantum_sim_lib GTest::gtest_main)
add_test(NAME NoiseTest COMMAND test_noise)

# Density matrix tests
add_executable(test_density_matrix tests/test_density_matrix.cu)
target_link_libraries(test_density_matrix quantum_sim_lib GTest::gtest_main)
add_test(NAME DensityMatrixTest COMMAND test_density_matrix)

# ============================================================================
# Benchmarks
# ============================================================================
add_executable(benchmark_gates benchmarks/benchmark_gates.cu)
target_link_libraries(benchmark_gates quantum_sim_lib cudart)

add_executable(benchmark_scaling benchmarks/benchmark_scaling.cu)
target_link_libraries(benchmark_scaling quantum_sim_lib cudart)

add_executable(benchmark_hadamard benchmarks/benchmark_hadamard.cu)
target_link_libraries(benchmark_hadamard quantum_sim_lib cudart)

add_executable(benchmark_optimized benchmarks/benchmark_optimized.cu)
target_link_libraries(benchmark_optimized quantum_sim_lib cudart)

# cuStateVec comparison benchmark (requires cuQuantum SDK)
# Look for cuStateVec in pip-installed location
set(CUQUANTUM_ROOT "$ENV{HOME}/.local/lib/python3.12/site-packages/cuquantum")
if(EXISTS "${CUQUANTUM_ROOT}/include/custatevec.h")
    message(STATUS "Found cuQuantum at: ${CUQUANTUM_ROOT}")
    
    add_executable(benchmark_custatevec benchmarks/benchmark_custatevec.cu)
    target_include_directories(benchmark_custatevec PRIVATE ${CUQUANTUM_ROOT}/include)
    target_link_directories(benchmark_custatevec PRIVATE ${CUQUANTUM_ROOT}/lib)
    target_link_libraries(benchmark_custatevec quantum_sim_lib cudart custatevec)
    
    # Set RPATH so the executable can find the library at runtime
    set_target_properties(benchmark_custatevec PROPERTIES
        BUILD_RPATH "${CUQUANTUM_ROOT}/lib"
        INSTALL_RPATH "${CUQUANTUM_ROOT}/lib"
    )
else()
    message(STATUS "cuQuantum not found - skipping benchmark_custatevec")
endif()
