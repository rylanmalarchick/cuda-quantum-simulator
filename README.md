# CUDA Quantum Simulator

A high-performance quantum state vector simulator implemented in CUDA C++. This project demonstrates GPU-accelerated quantum computing simulation, targeting research-grade code quality suitable for PhD program portfolios.

## Features

- **GPU-Accelerated Simulation**: Full state vector simulation on NVIDIA GPUs
- **Standard Gate Set**: X, Y, Z, H, S, T, S†, T†, Rx, Ry, Rz, CNOT, CZ, SWAP, Toffoli (CCX), CRY, CRZ
- **Noise Models**: Depolarizing, amplitude damping (T1), phase damping (T2), bit flip, phase flip
- **Batched Simulation**: Run multiple trajectories in parallel for Monte Carlo sampling
- **Measurement**: Single-qubit measurement with state collapse and multi-shot sampling
- **Fluent Circuit API**: Chainable gate operations with input validation
- **RAII Memory Management**: Safe GPU memory handling with automatic cleanup
- **Validated Against Industry Standards**: Cross-validated with Qiskit and Cirq
- **Comprehensive Test Suite**: Unit tests using Google Test framework

## Benchmark Results

Tested on NVIDIA RTX 4070 Laptop GPU (8GB VRAM, Compute Capability 8.9)

### GPU vs CPU Performance

![GPU vs CPU Comparison](docs/images/gpu_vs_cpu_comparison.png)

The GPU simulator becomes faster than single-threaded CPU at **20 qubits**, achieving:
- **1.1x speedup** at 20 qubits (1M states)
- **4.1x speedup** at 22 qubits (4M states)

### Scaling Characteristics

![Qubit Scaling](docs/images/qubit_scaling.png)

| Qubits | States | Memory | 100 H Gates |
|--------|--------|--------|-------------|
| 20 | 1M | 16 MB | 148 ms |
| 22 | 4M | 64 MB | 194 ms |
| 24 | 16M | 256 MB | 516 ms |
| 26 | 67M | 1 GB | 1.66 s |

### Gate Throughput (20 qubits)

![Gate Throughput](docs/images/gate_throughput.png)

| Gate | Throughput |
|------|------------|
| CNOT | 53,200 gates/s |
| X | 34,300 gates/s |
| H | 24,600 gates/s |
| Rz | 5,940 gates/s |

## Requirements

- **CUDA Toolkit** 12.0 or later
- **CMake** 3.18 or later
- **C++17** compatible compiler
- **NVIDIA GPU** with Compute Capability 7.0+ (tested on RTX 4070)

## Building

```bash
# Clone and build
cd cuda-quantum-simulator
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Run demo
./quantum_sim

# Run benchmarks
./benchmark_gates
./benchmark_scaling
```

## Usage

### Basic Circuit Execution

```cpp
#include "Simulator.hpp"
using namespace qsim;

// Create a 2-qubit Bell state
Simulator sim(2);
Circuit circuit(2);
circuit.h(0).cnot(0, 1);  // |Φ+⟩ = (|00⟩ + |11⟩)/√2

sim.run(circuit);
auto probs = sim.getProbabilities();
// probs[0] ≈ 0.5, probs[3] ≈ 0.5
```

### Measurement and Sampling

```cpp
// Sample from the probability distribution (non-destructive)
auto samples = sim.sample(1000);

// Measure a single qubit (collapses state)
int result = sim.measureQubit(0);
```

### Factory Circuits

```cpp
// Pre-built circuits for common states
auto bell = Circuit::createBellCircuit();
auto ghz = Circuit::createGHZCircuit(5);
auto random = Circuit::createRandomCircuit(4, 20);  // 4 qubits, depth 20
```

### Noisy Simulation

```cpp
#include "NoiseModel.cuh"
using namespace qsim;

// Create a noise model with depolarizing error
NoiseModel noise;
noise.addDepolarizing(0.01);  // 1% depolarizing probability per gate

// Run noisy simulation (Monte Carlo wavefunction method)
NoisySimulator sim(3, noise, /*seed=*/42);
Circuit circuit(3);
circuit.h(0).cnot(0, 1).cnot(1, 2);

sim.run(circuit);
auto probs = sim.getProbabilities();
```

### Batched Monte Carlo Sampling

```cpp
// Run 1000 noisy trajectories in parallel on GPU
NoiseModel noise;
noise.addDepolarizing(0.005);
noise.addAmplitudeDamping(0.001);  // T1 decay

BatchedSimulator batch(/*n_qubits=*/3, /*batch_size=*/1000, noise);
Circuit circuit(3);
circuit.h(0).cnot(0, 1).cnot(1, 2);

batch.run(circuit);

// Get average probabilities across all trajectories
auto avg_probs = batch.getAverageProbabilities();

// Get histogram of measurement outcomes
auto histogram = batch.getHistogram();  // Counts for each basis state
```

### Available Noise Channels

| Noise Type | Description | Usage |
|------------|-------------|-------|
| Depolarizing | Random Pauli error (X, Y, or Z with equal probability) | `noise.addDepolarizing(p)` |
| Amplitude Damping | T1 decay - relaxation to ground state | `noise.addAmplitudeDamping(gamma)` |
| Phase Damping | T2 dephasing - loss of phase coherence | `noise.addPhaseDamping(gamma)` |
| Bit Flip | X error with probability p | `noise.addBitFlip(p)` |
| Phase Flip | Z error with probability p | `noise.addPhaseFlip(p)` |
| Bit-Phase Flip | Y error with probability p | `noise.addBitPhaseFlip(p)` |

## Architecture

### Core Components

```
include/
├── StateVector.cuh    # GPU-resident quantum state (2^n complex amplitudes)
├── Gates.cuh          # CUDA kernels for quantum gates
├── Circuit.hpp        # Circuit representation with fluent API
├── Simulator.hpp      # GPU simulator orchestration
├── NoiseModel.cuh     # Noise models and batched simulation
├── CudaMemory.cuh     # RAII wrapper for CUDA memory
└── Constants.hpp      # Configuration and math constants

src/
├── StateVector.cu     # State vector implementation
├── Gates.cu           # Gate kernel implementations
├── Circuit.cpp        # Circuit builder
├── Simulator.cu       # GPU and CPU simulator implementations
├── NoiseModel.cu      # Noise model and batched simulator
└── main.cpp           # Demo executable
```

### Qubit Ordering Convention

This simulator uses **big-endian** (most significant qubit first) ordering, matching Cirq's default:

- Qubit 0 is the **most significant bit** (leftmost in ket notation)
- For a 3-qubit state |q0 q1 q2⟩: `index = q0×4 + q1×2 + q2×1`

Example for 3 qubits:
```
Index:  0     1     2     3     4     5     6     7
State: |000⟩ |001⟩ |010⟩ |011⟩ |100⟩ |101⟩ |110⟩ |111⟩
```

## Validation

Cross-validated against Qiskit and Cirq for correctness:

```bash
# Set up Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r validation/requirements.txt

# Run validation
python validation/validate_against_qiskit.py  # All tests pass
python validation/validate_against_cirq.py    # All tests pass
```

Validation covers:
- Bell state and GHZ states
- All single-qubit gates (X, Y, Z, H, S, T, Rx, Ry, Rz)
- All two-qubit gates (CNOT, CZ, SWAP)
- Random deep circuits (20+ gates)

## Testing

The test suite uses Google Test and covers:

| Test Suite | Description |
|------------|-------------|
| `test_statevector` | State initialization, normalization, measurement |
| `test_gates` | Gate correctness and matrix properties |
| `test_gate_algebra` | Gate identities (HH=I, XX=I, etc.) |
| `test_gpu_cpu_equivalence` | GPU matches CPU reference |
| `test_boundary` | Edge cases and error handling |
| `test_noise` | Noise models and batched simulation |

```bash
# Run all tests
cd build && ctest --output-on-failure

# Run specific test suite
./test_statevector --gtest_filter='*Measurement*'
```

## Technical Details

### Memory Requirements

State vector size = 2^n × 16 bytes (double precision complex):
- 20 qubits: 16 MB
- 25 qubits: 512 MB  
- 27 qubits: 2 GB (practical limit for 8GB GPU)

### Performance Characteristics

- **Kernel launch overhead**: ~130ms for small qubit counts dominates execution
- **Memory bandwidth bound**: Performance scales with state vector size at high qubit counts
- **GPU advantage**: Emerges at 20+ qubits where parallelism outweighs overhead

### Design Decisions

1. **Double precision only**: Ensures numerical stability for deep circuits
2. **Synchronous execution**: Simpler debugging, consistent timing
3. **Big-endian ordering**: Matches Cirq; Qiskit comparison requires bit reversal
4. **RAII everywhere**: No manual cudaFree calls, exception-safe

## Future Work

- [ ] Density matrix simulation for mixed states
- [ ] Shared memory optimization for gate kernels
- [ ] Coalesced memory access patterns
- [ ] cuStateVec integration for comparison
- [ ] Multi-GPU support

## License

MIT License - see LICENSE file for details.

## Author

Built as a portfolio project demonstrating GPU programming, quantum computing fundamentals, and research-quality software engineering.
