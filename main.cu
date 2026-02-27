#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                       \
    cudaError_t err__ = (call);                                     \
    if (err__ != cudaSuccess) {                                     \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                     __FILE__, __LINE__, cudaGetErrorString(err__)); \
        std::exit(1);                                               \
    }                                                               \
} while (0)

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

    const int N= 16;
    //1. Creating points on CPU
    std::vector<float3> h_points = make_random_points(N);
    //2. Allocate GPU buffers
    float3* d_points = nullptr;
    float* d_dist2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, N*sizeof(float3)));
    CUDA_CHECK(cudaMalloc((&d_dist2), N*sizeof(float)));
    //3. Copy from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(),
        N*sizeof(float3), cudaMemcpyHostToDevice));
    //4. Launch kernel
    dim3 block(256);
    dim3 grid((N+block.x -1)/ block.x);
    dist_to_first<<<grid, block>>>(N, d_points, d_dist2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    //5. Copy from GPU to CPU and print
    std::vector<float> h_dist2(N);
    CUDA_CHECK(cudaMemcpy(h_dist2.data(), d_dist2, N*sizeof(float),
        cudaMemcpyDeviceToHost));
    std::printf("points[0] = (%.3f, %.3f, %.3f)\n",
            h_points[0].x, h_points[0].y, h_points[0].z);

    for (int i = 0; i < N; i++) {
        std::printf("i=%2d  p=(%.3f, %.3f, %.3f)  dist2_to_p0=%.6f\n",
                    i, h_points[i].x, h_points[i].y, h_points[i].z, h_dist2[i]);
    }
    //6. Cleanup
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_dist2));

    return 0;
}