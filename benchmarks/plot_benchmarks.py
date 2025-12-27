#!/usr/bin/env python3
"""
Benchmark Visualization for CUDA Quantum Simulator

Generates publication-quality plots from benchmark data.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11


def plot_gpu_vs_cpu():
    """Plot GPU vs CPU speedup comparison."""
    # Data from benchmark_scaling output
    qubits = np.array([10, 12, 14, 16, 18, 20, 22])
    gpu_ms = np.array([163.15, 160.17, 160.53, 161.11, 162.87, 176.22, 227.26])
    cpu_ms = np.array([0.19, 0.70, 2.95, 12.64, 46.84, 185.90, 927.58])
    speedup = cpu_ms / gpu_ms
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Execution time comparison
    ax1.semilogy(qubits, gpu_ms, 'o-', label='GPU (RTX 4070)', linewidth=2, markersize=8)
    ax1.semilogy(qubits, cpu_ms, 's--', label='CPU (single-thread)', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('100 Hadamard Gates: GPU vs CPU')
    ax1.legend()
    ax1.set_xticks(qubits)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Speedup
    colors = ['#2ecc71' if s >= 1 else '#e74c3c' for s in speedup]
    bars = ax2.bar(qubits, speedup, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Break-even')
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('Speedup (CPU time / GPU time)')
    ax2.set_title('GPU Speedup Over CPU')
    ax2.set_xticks(qubits)
    ax2.legend()
    
    # Add value labels on bars
    for bar, spd in zip(bars, speedup):
        height = bar.get_height()
        ax2.annotate(f'{spd:.1f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('gpu_vs_cpu_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: gpu_vs_cpu_comparison.png")
    plt.close()


def plot_qubit_scaling():
    """Plot how execution time scales with qubit count."""
    # Data from benchmark_scaling output
    qubits = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26])
    gate_time_ms = np.array([135.76, 132.65, 133.56, 133.56, 136.38, 147.96, 193.64, 515.83, 1659.42])
    memory_mb = np.array([0.0, 0.1, 0.2, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Gate execution time
    ax1.semilogy(qubits, gate_time_ms, 'o-', color='#3498db', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Time for 100 H Gates (ms)')
    ax1.set_title('GPU Gate Execution Scaling')
    ax1.set_xticks(qubits)
    ax1.grid(True, alpha=0.3)
    
    # Annotate the knee point
    ax1.axvline(x=22, color='#e74c3c', linestyle='--', alpha=0.7, label='Performance inflection')
    ax1.legend()
    
    # Right plot: Memory usage
    ax2.semilogy(qubits, memory_mb, 's-', color='#9b59b6', linewidth=2, markersize=8)
    ax2.axhline(y=8192, color='#e74c3c', linestyle='--', label='GPU Memory (8 GB)', alpha=0.7)
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('State Vector Size (MB)')
    ax2.set_title('Memory Requirements')
    ax2.set_xticks(qubits)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qubit_scaling.png', dpi=150, bbox_inches='tight')
    print("Saved: qubit_scaling.png")
    plt.close()


def plot_gate_throughput():
    """Plot gate throughput for different gate types."""
    # Data from benchmark_gates output (20 qubits as representative)
    gates = ['H', 'X', 'Rz', 'CNOT']
    throughput_20q = np.array([24600, 34300, 5940, 53200])  # gates/s at 20 qubits
    throughput_15q = np.array([124000, 139000, 79300, 159000])  # gates/s at 15 qubits
    
    x = np.arange(len(gates))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, throughput_15q / 1000, width, label='15 qubits', color='#3498db')
    bars2 = ax.bar(x + width/2, throughput_20q / 1000, width, label='20 qubits', color='#e74c3c')
    
    ax.set_xlabel('Gate Type')
    ax.set_ylabel('Throughput (k gates/s)')
    ax.set_title('Gate Throughput by Type and Qubit Count')
    ax.set_xticks(x)
    ax.set_xticklabels(gates)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}k',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}k',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('gate_throughput.png', dpi=150, bbox_inches='tight')
    print("Saved: gate_throughput.png")
    plt.close()


def plot_summary():
    """Create a single summary figure with key results."""
    fig = plt.figure(figsize=(16, 10))
    
    # GPU vs CPU comparison (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    qubits = np.array([10, 12, 14, 16, 18, 20, 22])
    gpu_ms = np.array([163.15, 160.17, 160.53, 161.11, 162.87, 176.22, 227.26])
    cpu_ms = np.array([0.19, 0.70, 2.95, 12.64, 46.84, 185.90, 927.58])
    
    ax1.semilogy(qubits, gpu_ms, 'o-', label='GPU', linewidth=2, markersize=7)
    ax1.semilogy(qubits, cpu_ms, 's--', label='CPU', linewidth=2, markersize=7)
    ax1.set_xlabel('Qubits')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('GPU vs CPU Performance')
    ax1.legend()
    ax1.set_xticks(qubits)
    
    # Speedup (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    speedup = cpu_ms / gpu_ms
    colors = ['#2ecc71' if s >= 1 else '#e74c3c' for s in speedup]
    ax2.bar(qubits, speedup, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=1, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Qubits')
    ax2.set_ylabel('Speedup')
    ax2.set_title('GPU Speedup Factor')
    ax2.set_xticks(qubits)
    
    # Qubit scaling (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    all_qubits = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26])
    gate_time = np.array([135.76, 132.65, 133.56, 133.56, 136.38, 147.96, 193.64, 515.83, 1659.42])
    ax3.semilogy(all_qubits, gate_time, 'o-', color='#3498db', linewidth=2, markersize=7)
    ax3.set_xlabel('Qubits')
    ax3.set_ylabel('Time for 100 H gates (ms)')
    ax3.set_title('GPU Scaling')
    ax3.set_xticks(all_qubits)
    
    # Gate throughput (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    gates = ['H', 'X', 'Rz', 'CNOT']
    throughput = np.array([24600, 34300, 5940, 53200]) / 1000
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']
    ax4.bar(gates, throughput, color=colors, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Gate Type')
    ax4.set_ylabel('Throughput (k gates/s)')
    ax4.set_title('Gate Throughput @ 20 Qubits')
    
    plt.suptitle('CUDA Quantum Simulator Benchmarks (RTX 4070 Laptop GPU)', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig('benchmark_summary.png', dpi=150, bbox_inches='tight')
    print("Saved: benchmark_summary.png")
    plt.close()


if __name__ == '__main__':
    print("Generating benchmark plots...")
    plot_gpu_vs_cpu()
    plot_qubit_scaling()
    plot_gate_throughput()
    plot_summary()
    print("\nAll plots generated successfully!")
