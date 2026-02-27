#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cfloat>

#include <cuda_runtime.h>

#include "knn_cpu.h"

#define CUDA_CHECK(call) do {                                       \
    cudaError_t err__ = (call);                                     \
    if (err__ != cudaSuccess) {                                     \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                     __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1);                                               \
    }                                                               \
} while (0)


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

int main() {
    // Uncomment below for Step 0:
    // hello_kernel<<<2, 4>>>(); //defining 2 blocks and 4 threads
    // // To catch launch errors
    // CUDA_CHECK(cudaGetLastError());
    // //Here we wait for the kernel to complete
    // CUDA_CHECK(cudaDeviceSynchronize());
    // return 0;

    const int N = 256;
    //1. Creating points on CPU
    std::vector<float3> h_points = make_random_points(N);
    //CPU reference
    std::vector<float> h_ref = knn3_mean_dist2_cpu(h_points);

    //2. Allocate GPU buffers
    float3* d_points = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, N*sizeof(float3)));
    CUDA_CHECK(cudaMalloc((&d_out), N*sizeof(float)));

    //3. Copy from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(),
        N*sizeof(float3), cudaMemcpyHostToDevice));

    //4. Launch kernel
    dim3 block(256);
    dim3 grid((N+block.x -1)/ block.x);
    dist_to_first<<<grid, block>>>(N, d_points, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    //5. Copy from GPU to CPU and print
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

    // 6. Compare CPU result to GPU
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
    for (int i = 0; i < 5; i++) {
        std::printf("  i=%d  ref=%.6f  gpu=%.6f\n", i, h_ref[i], h_gpu[i]);
    }
    //7. Cleanup
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}