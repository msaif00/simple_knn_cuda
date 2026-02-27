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

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cfloat>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "knn_cpu.h"

#define CUDA_CHECK(call) do {                                       \
    cudaError_t err__ = (call);                                     \
    if (err__ != cudaSuccess) {                                     \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                     __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1);                                               \
    }                                                               \
} while (0)


__host__ __device__ static inline uint32_t prepMorton(uint32_t x) {
    // Spread 10 bits to 30 bits by inserting 2 zeros between each bit.
    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x <<  8)) & 0x0300F00Fu;
    x = (x | (x <<  4)) & 0x030C30C3u;
    x = (x | (x <<  2)) & 0x09249249u;
    return x;
}

__host__ __device__ static inline uint32_t coord2Morton(float3 p, float3 bbMin, float3 bbMax) {
    // Normalize to [0, 1023] (10 bits) per axis.
    // Note: This assumes bbMax > bbMin on all axes.
    float nx = (p.x - bbMin.x) / (bbMax.x - bbMin.x);
    float ny = (p.y - bbMin.y) / (bbMax.y - bbMin.y);
    float nz = (p.z - bbMin.z) / (bbMax.z - bbMin.z);

    nx = fminf(fmaxf(nx, 0.0f), 1.0f);
    ny = fminf(fmaxf(ny, 0.0f), 1.0f);
    nz = fminf(fmaxf(nz, 0.0f), 1.0f);

    const uint32_t ix = (uint32_t)(nx * 1023.0f);
    const uint32_t iy = (uint32_t)(ny * 1023.0f);
    const uint32_t iz = (uint32_t)(nz * 1023.0f);

    const uint32_t x = prepMorton(ix);
    const uint32_t y = prepMorton(iy);
    const uint32_t z = prepMorton(iz);

    // Interleave bits: x in bit 0, y in bit 1, z in bit 2, repeating...
    return x | (y << 1) | (z << 2);
}

__global__ void morton_kernel(int n, const float3* points, float3 bbMin, float3 bbMax, uint32_t*outCodes) {

    int idx = blockIdx.x * blockDim.x+threadIdx.x;
    if (idx >= n) return;
    outCodes[idx] = coord2Morton(points[idx], bbMin, bbMax);
}

__device__ __forceinline__ void updateKBest3(float dist2, float best[3]) {

    for (int j=0; j<3; j++) {
        if (dist2 < best[j]) {
            float t = best[j];
            best[j] = dist2;
            dist2 = t;
        }
    }
}

// Brute-force KNN3 mean distance^2: O(N^2)
__global__ void knn3_mean_dist2_gpu(int n, const float3* points, float* outMeanDist2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3 pi = points[i];
    float best[3] = {FLT_MAX, FLT_MAX, FLT_MAX};

    for (int j = 0; j < n; j++) {
        if (j == i) continue;

        float3 pj = points[j];
        float dx = pi.x - pj.x;
        float dy = pi.y - pj.y;
        float dz = pi.z - pj.z;
        float dist2 = dx*dx + dy*dy + dz*dz;

        updateKBest3(dist2, best);
    }

    outMeanDist2[i] = (best[0] + best[1] + best[2]) / 3.0f;
}

__global__ void hello_kernel() {
    int globalThread =  blockIdx.x + blockDim.x * threadIdx.x;
    printf("Hello from RTX 2080 GPU, thread %d in block %d at global %d!\n",
        threadIdx.x, blockIdx.x, globalThread);
}

// Compute squared distance from each point to point[0]
__global__ void dist_to_first(int n, const float3* points, float* outDist2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float3 p0 = points[0];
    float3 p = points[idx];

    float dx = p.x - p0.x;
    float dy = p.y - p0.y;
    float dz = p.z - p0.z;

    outDist2[idx] = dx*dx + dy*dy + dz*dz;
}

static std::vector<float3> make_random_points(int n, unsigned seed=123) {

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<float3> pts(n);
    for (int i = 0; i < n; i++) {
        pts[i] = make_float3(uni(rng), uni(rng), uni(rng));
    }
    return pts;
}

static void compute_bbox_cpu(const std::vector<float3>&pts, float3& outMin, float3& outMax) {
    outMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    outMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (const auto& p : pts) {
        outMin.x = std::min(outMin.x,p.x);
        outMin.y = std::min(outMin.y,p.y);
        outMin.z = std::min(outMin.z,p.z);

        outMax.x = std::max(outMax.x,p.x);
        outMax.y = std::max(outMax.y,p.y);
        outMax.z = std::max(outMax.z,p.z);
    }

    //Avoid divison by zero
    const float eps = 1e-6f;
    if ((outMax.x - outMin.x) < eps) outMax.x = outMin.x + eps;
    if ((outMax.y - outMin.y) < eps) outMax.y = outMin.y + eps;
    if ((outMax.z - outMin.z) < eps) outMax.z = outMin.z + eps;
}


int main() {
    // Uncomment below for Step 0:
    // hello_kernel<<<2, 4>>>(); //defining 2 blocks and 4 threads
    // // To catch launch errors
    // CUDA_CHECK(cudaGetLastError());
    // //Here we wait for the kernel to complete
    // CUDA_CHECK(cudaDeviceSynchronize());
    // return 0;

    //Testing known points
    std::vector<float3> h_points = {
        make_float3(0.0f, 0.0f, 0.0f),
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f),
        make_float3(0.0f, 0.0f, 1.0f),
        make_float3(1.0f, 1.0f, 0.0f),
    };

    const int N = static_cast<int>(h_points.size());
    // const int N = 256;
    //1. Creating points on CPU
    // std::vector<float3> h_points = make_random_points(N);
    //CPU reference
    std::vector<float> h_ref = knn3_mean_dist2_cpu(h_points);

    //2. Allocate GPU buffers
    float3* d_points = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, N*sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_out, N*sizeof(float)));

    //3. Copy from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(),
        N*sizeof(float3), cudaMemcpyHostToDevice));

    //5. Compute Morton codes w.bbox on CPU
    float3 bbMin, bbMax;
    compute_bbox_cpu(h_points, bbMin, bbMax);

    thrust::device_vector<uint32_t> d_codes(N);
    dim3 block(256);
    dim3 grid((N+block.x -1)/ block.x);
    //dist_to_first<<<grid, block>>>(N, d_points, d_out);
    morton_kernel<<<grid, block>>>(N, d_points, bbMin, bbMax, thrust::raw_pointer_cast(d_codes.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Creaate indices and sort them by Morton code
    thrust::device_vector<uint32_t> d_indices(N);
    thrust::sequence(d_indices.begin(), d_indices.end());
    thrust::sort_by_key(d_codes.begin(),d_codes.end(), d_indices.begin());

    // As a check print the first few indices
    std::vector<uint32_t> h_indices(10);
    CUDA_CHECK(cudaMemcpy(h_indices.data(),
        thrust::raw_pointer_cast(d_indices.data()),
            10*sizeof(uint32_t),
            cudaMemcpyDeviceToHost));

    // std::printf("First 10 indices after Morton sort:\n");
    // for (int i=0; i<10; i++) std::printf("%d ", h_indices[i]);

    //6. Run Brute=force KNN on GPU
    knn3_mean_dist2_gpu<<<grid, block>>>(N, d_points, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    //7. Copy from GPU to CPU and print
    std::vector<float> h_gpu(N);
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, N*sizeof(float),
        cudaMemcpyDeviceToHost));
    // std::printf("points[0] = (%.3f, %.3f, %.3f)\n",
    //         h_points[0].x, h_points[0].y, h_points[0].z);
    //
    // for (int i = 0; i < N; i++) {
    //     std::printf("i=%2d  p=(%.3f, %.3f, %.3f)  dist2_to_p0=%.6f\n",
    //                 i, h_points[i].x, h_points[i].y, h_points[i].z, h_gpu[i]);
    // }

    // 8. Compare CPU result to GPU
    float maxAbsErr = 0.0f;
    int maxIdx = -1;
    for (int i =0; i<N; i++) {
        float err = std::fabs(h_gpu[i] - h_ref[i]);
        if (err > maxAbsErr) {
            maxAbsErr = err;
            maxIdx = i;
        }
    }
    std::printf("KNN3 mean dist^2 comparison:\n");
    std::printf("  N=%d\n", N);
    std::printf("  maxAbsErr = %.9g at i=%d\n", maxAbsErr, maxIdx);
    if (maxIdx >= 0) {
        std::printf("  ref[%d]=%.9g  gpu[%d]=%.9g\n",
                    maxIdx, h_ref[maxIdx], maxIdx, h_gpu[maxIdx]);
    }

    // Print a few sample outputs (sanity)
    // for (int i = 0; i < 5; i++) {
    //     std::printf("  i=%d  ref=%.6f  gpu=%.6f\n", i, h_ref[i], h_gpu[i]);
    // }
    //9. Cleanup
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}