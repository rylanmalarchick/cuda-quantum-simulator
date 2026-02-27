#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Rylan Malarchick
#
# Run all 9 test suites under valgrind to verify zero memory leaks.
#
# CUDA programs produce known false positives (ioctl noise, context memory)
# that are suppressed via cuda.supp.  The check that matters is:
#   definitely lost = 0 bytes  (no application-level leaks)
#   indirectly lost = 0 bytes
#
# Usage:
#   ./valgrind.sh            # from project root (builds first if needed)
#   ./valgrind.sh --no-build # skip build step

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
SUPP="$SCRIPT_DIR/cuda.supp"

VALGRIND_FLAGS=(
    --suppressions="$SUPP"
    --leak-check=full
    --show-leak-kinds=definite,indirect
    --errors-for-leak-kinds=definite,indirect
    --error-exitcode=1
)

TESTS=(
    test_warmup
    test_statevector
    test_gates
    test_gate_algebra
    test_gpu_cpu_equivalence
    test_boundary
    test_noise
    test_density_matrix
    test_optimized_gates
)

# Build if requested (default: build)
if [[ "${1:-}" != "--no-build" ]]; then
    echo "=== Building ==="
    cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -Wno-dev -q
    make -C "$BUILD_DIR" -j"$(nproc)" --no-print-directory
    echo ""
fi

echo "=== Valgrind Memory Check (9 test suites) ==="
echo "Suppression file: $SUPP"
echo ""

PASS=0
FAIL=0

for test in "${TESTS[@]}"; do
    printf "  %-36s" "$test"
    output=$(valgrind "${VALGRIND_FLAGS[@]}" "$BUILD_DIR/$test" 2>&1)
    def_lost=$(echo "$output" | grep "definitely lost:" | awk '{print $4}')
    ind_lost=$(echo "$output" | grep "indirectly lost:" | awk '{print $4}')
    errors=$(echo "$output" | grep "ERROR SUMMARY:" | awk '{print $4}')

    if [[ "$errors" == "0" && "$def_lost" == "0" && "$ind_lost" == "0" ]]; then
        echo "CLEAN  (definitely lost: 0, indirectly lost: 0)"
        ((PASS++)) || true
    else
        echo "FAIL"
        echo "$output" | grep -E "ERROR SUMMARY|definitely lost|indirectly lost"
        ((FAIL++)) || true
    fi
done

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
echo "All tests: definitely lost = 0 bytes  (clean valgrind)"
