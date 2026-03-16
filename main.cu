/* Original Code from GRAPHDECO, re-implemented by MSP 2026
* Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>

#include <cuda_runtime.h>

#include "knn_cpu.h"
#include "simple_knn.h"

#define CUDA_CHECK(call) do {                                        \
    cudaError_t err__ = (call);                                      \
    if (err__ != cudaSuccess) {                                      \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                     __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1);                                                \
    }                                                                \
} while (0)

static std::vector<float3> make_random_points(int n, unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float3> pts(n);
    for (int i = 0; i < n; i++)
        pts[i] = make_float3(dist(rng), dist(rng), dist(rng));
    return pts;
}

int main()
{
    const int N = 4096;  // Try 16384, 65536, etc.

    std::printf("SimpleKNN test with N=%d points\n\n", N);

    // 1) Generate random points
    std::vector<float3> h_points = make_random_points(N);

    // 2) CPU reference (exact brute-force)
    std::printf("Computing CPU reference (brute-force)...\n");
    std::vector<float> h_ref = knn3_mean_dist2_cpu(h_points);

    // 3) Allocate GPU memory
    float3* d_points = nullptr;
    float*  d_dists  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_dists,  N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), N * sizeof(float3), cudaMemcpyHostToDevice));

    // 4) Run our SimpleKNN implementation
    std::printf("Running GPU SimpleKNN::knn()...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    SimpleKNN::knn(N, d_points, d_dists);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpuTimeMs = 0.0f;
    cudaEventElapsedTime(&gpuTimeMs, start, stop);
    std::printf("GPU time: %.3f ms\n\n", gpuTimeMs);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 5) Copy results back
    std::vector<float> h_gpu(N);
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_dists, N * sizeof(float), cudaMemcpyDeviceToHost));

    // 6) Compare GPU vs CPU reference
    float maxAbsErr = 0.0f;
    int   maxIdx = -1;
    int   exactMatches = 0;

    for (int i = 0; i < N; i++) {
        float err = std::fabs(h_gpu[i] - h_ref[i]);
        if (err < 1e-9f) exactMatches++;
        if (err > maxAbsErr) {
            maxAbsErr = err;
            maxIdx = i;
        }
    }

    std::printf("=== Comparison: GPU vs CPU reference ===\n");
    std::printf("  Total points:   %d\n", N);
    std::printf("  Exact matches:  %d / %d\n", exactMatches, N);
    std::printf("  Max abs error:  %.9g", maxAbsErr);
    if (maxIdx >= 0)
        std::printf(" (at i=%d: cpu=%.9g gpu=%.9g)", maxIdx, h_ref[maxIdx], h_gpu[maxIdx]);
    std::printf("\n\n");

    // Show a few samples
    std::printf("Sample outputs:\n");
    for (int i = 0; i < 10; i++) {
        std::printf("  i=%d  cpu=%.6f  gpu=%.6f\n", i, h_ref[i], h_gpu[i]);
    }

    // 7) Cleanup
    CUDA_CHECK(cudaFree(d_dists));
    CUDA_CHECK(cudaFree(d_points));

    std::printf("\nDone.\n");
    return 0;
}