#include "simple_knn.h"

#include <cstdint>
#include <cfloat>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

// ============================================================================
// Configuration
// ============================================================================

#define BOX_SIZE 1024

// ============================================================================
// CUB custom reduction operators for float3 min/max
// ============================================================================


struct Float3Min {
    __device__ __forceinline__
    float3 operator()(const float3& a, const float3& b) const {
        return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
    }
};

struct Float3Max {
    __device__ __forceinline__
    float3 operator()(const float3& a, const float3& b) const {
        return make_float3(fmaxf(a.x,b.x), fmaxf(a.x, b.y), fmaxf(a.x,b.y));
    }
};

// ============================================================================
// Morton code encoding
// ============================================================================


__host__ __device__ static inline uint32_t prepMorton(uint32_t x)
{
    // Spread 10 bits across 30 bits (insert 2 zeros between each bit)
    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x <<  8)) & 0x0300F00Fu;
    x = (x | (x <<  4)) & 0x030C30C3u;
    x = (x | (x <<  2)) & 0x09249249u;
    return x;
}

__host__ __device__ static inline uint32_t coord2Morton(float3 p, float3 bbMin, float3 bbMax)
{
    // Normalize each axis to [0, 1], then quantize to 10 bits [0, 1023]
    float nx = (p.x - bbMin.x) / (bbMax.x - bbMin.x);
    float ny = (p.y - bbMin.y) / (bbMax.y - bbMin.y);
    float nz = (p.z - bbMin.z) / (bbMax.z - bbMin.z);

    // Clamp to [0, 1] for safety
    nx = fminf(fmaxf(nx, 0.0f), 1.0f);
    ny = fminf(fmaxf(ny, 0.0f), 1.0f);
    nz = fminf(fmaxf(nz, 0.0f), 1.0f);

    const uint32_t ix = static_cast<uint32_t>(nx * 1023.0f);
    const uint32_t iy = static_cast<uint32_t>(ny * 1023.0f);
    const uint32_t iz = static_cast<uint32_t>(nz * 1023.0f);

    // Interleave: x in bit positions 0,3,6,...  y in 1,4,7,...  z in 2,5,8,...
    return prepMorton(ix) | (prepMorton(iy) << 1) | (prepMorton(iz) << 2);
}

__global__ void computeMortonCodes(int n, const float3* points, float3 bbMin, float3 bbMax, uint32_t* codes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    codes[idx] = coord2Morton(points[idx], bbMin, bbMax);
}

// ============================================================================
// Box (AABB) data structure and helpers
// ============================================================================

struct MinMax
{
    float3 minn;
    float3 maxx;
};

__device__ __host__ inline float distBoxPoint(const MinMax& box, const float3& p)
{
    // Squared distance from point p to the closest point on/in the box.
    // Returns 0 if p is inside the box.
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;

    if (p.x < box.minn.x)       dx = box.minn.x - p.x;
    else if (p.x > box.maxx.x)  dx = p.x - box.maxx.x;

    if (p.y < box.minn.y)       dy = box.minn.y - p.y;
    else if (p.y > box.maxx.y)  dy = p.y - box.maxx.y;

    if (p.z < box.minn.z)       dz = box.minn.z - p.z;
    else if (p.z > box.maxx.z)  dz = p.z - box.maxx.z;

    return dx*dx + dy*dy + dz*dz;
}

// ============================================================================
// Kernel: compute AABB of each box using shared-memory tree reduction
// Launch: <<<numBoxes, BOX_SIZE>>>
// ============================================================================

__global__ void boxMinMaxKernel(
    uint32_t n,
    const float3* __restrict__ points,
    const uint32_t* __restrict__ sortedIndices,
    MinMax* __restrict__ boxes)
{
    const uint32_t globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load one point per thread, or a sentinel if past the end
    MinMax me;
    if (globalIdx < n) {
        float3 p = points[sortedIndices[globalIdx]];
        me.minn = p;
        me.maxx = p;
    } else {
        me.minn = make_float3( FLT_MAX,  FLT_MAX,  FLT_MAX);
        me.maxx = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }

    __shared__ MinMax smem[BOX_SIZE];

    // Tree reduction
    for (int offset = BOX_SIZE / 2; offset >= 1; offset /= 2) {
        if (threadIdx.x < 2 * offset)
            smem[threadIdx.x] = me;
        __syncthreads();

        if (threadIdx.x < offset) {
            MinMax other = smem[threadIdx.x + offset];
            me.minn.x = fminf(me.minn.x, other.minn.x);
            me.minn.y = fminf(me.minn.y, other.minn.y);
            me.minn.z = fminf(me.minn.z, other.minn.z);
            me.maxx.x = fmaxf(me.maxx.x, other.maxx.x);
            me.maxx.y = fmaxf(me.maxx.y, other.maxx.y);
            me.maxx.z = fmaxf(me.maxx.z, other.maxx.z);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        boxes[blockIdx.x] = me;
}

// ============================================================================
// Kernel: box-pruned exact 3-NN search
// Launch: <<<numBoxes, BOX_SIZE>>>
// ============================================================================

__global__ void knn3BoxPrunedKernel(
    uint32_t n,
    const float3* __restrict__ points,
    const uint32_t* __restrict__ sortedIndices,
    const MinMax* __restrict__ boxes,
    uint32_t numBoxes,
    float* __restrict__ outMeanDist2)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;  // sorted position
    if (s >= static_cast<int>(n)) return;

    const uint32_t i = sortedIndices[s];
    const float3 pi = points[i];

    float best[3] = {FLT_MAX, FLT_MAX, FLT_MAX};

    // --- Pass 1: local window to get an initial reject bound ---
    {
        const int lo = max(0, s - 3);
        const int hi = min(static_cast<int>(n) - 1, s + 3);
        for (int t = lo; t <= hi; t++) {
            if (t == s) continue;
            const uint32_t j = sortedIndices[t];
            const float3 pj = points[j];
            const float dx = pi.x - pj.x;
            const float dy = pi.y - pj.y;
            const float dz = pi.z - pj.z;
            float d2 = dx*dx + dy*dy + dz*dz;
            for (int k = 0; k < 3; k++) {
                if (d2 < best[k]) { float tmp = best[k]; best[k] = d2; d2 = tmp; }
            }
        }
    }

    const float reject = best[2];
    best[0] = FLT_MAX;
    best[1] = FLT_MAX;
    best[2] = FLT_MAX;

    // --- Pass 2: scan boxes with pruning ---
    for (uint32_t b = 0; b < numBoxes; b++) {
        float boxDist = distBoxPoint(boxes[b], pi);
        if (boxDist > reject || boxDist > best[2])
            continue;

        const int boxStart = b * BOX_SIZE;
        const int boxEnd   = min(static_cast<int>(n), static_cast<int>((b + 1) * BOX_SIZE));

        for (int t = boxStart; t < boxEnd; t++) {
            if (t == s) continue;
            const uint32_t j = sortedIndices[t];
            const float3 pj = points[j];
            const float dx = pi.x - pj.x;
            const float dy = pi.y - pj.y;
            const float dz = pi.z - pj.z;
            float d2 = dx*dx + dy*dy + dz*dz;
            for (int k = 0; k < 3; k++) {
                if (d2 < best[k]) { float tmp = best[k]; best[k] = d2; d2 = tmp; }
            }
        }
    }

    outMeanDist2[i] = (best[0] + best[1] + best[2]) / 3.0f;
}

// ============================================================================
// Public API: SimpleKNN::knn
// ============================================================================

void SimpleKNN::knn(int P, float3* points, float* meanDists)
{
    // ----- Step 1: Compute global bounding box on GPU using CUB -----
    float3* d_result = nullptr;
    cudaMalloc(&d_result, sizeof(float3));

    size_t tempBytes = 0;
    float3 identity = make_float3(0.0f, 0.0f, 0.0f);
    float3 bbMin, bbMax;

    // Query temp storage size for min reduction
    cub::DeviceReduce::Reduce(nullptr, tempBytes, points, d_result, P, Float3Min(), identity);
    thrust::device_vector<char> tempStorage(tempBytes);

    // Compute min
    cub::DeviceReduce::Reduce(tempStorage.data().get(), tempBytes, points, d_result, P, Float3Min(), identity);
    cudaMemcpy(&bbMin, d_result, sizeof(float3), cudaMemcpyDeviceToHost);

    // Compute max (reuse temp storage, query again in case size differs)
    cub::DeviceReduce::Reduce(nullptr, tempBytes, points, d_result, P, Float3Max(), identity);
    if (tempBytes > tempStorage.size()) tempStorage.resize(tempBytes);
    cub::DeviceReduce::Reduce(tempStorage.data().get(), tempBytes, points, d_result, P, Float3Max(), identity);
    cudaMemcpy(&bbMax, d_result, sizeof(float3), cudaMemcpyDeviceToHost);

    cudaFree(d_result);

    // Small epsilon to avoid divide-by-zero on flat axes
    const float eps = 1e-6f;
    if (bbMax.x - bbMin.x < eps) bbMax.x = bbMin.x + eps;
    if (bbMax.y - bbMin.y < eps) bbMax.y = bbMin.y + eps;
    if (bbMax.z - bbMin.z < eps) bbMax.z = bbMin.z + eps;

    // ----- Step 2: Compute Morton codes -----
    thrust::device_vector<uint32_t> mortonCodes(P);
    thrust::device_vector<uint32_t> mortonCodesSorted(P);

    dim3 block(256);
    dim3 grid((P + block.x - 1) / block.x);

    computeMortonCodes<<<grid, block>>>(P, points, bbMin, bbMax, mortonCodes.data().get());

    // ----- Step 3: Sort indices by Morton code using CUB radix sort -----
    thrust::device_vector<uint32_t> indices(P);
    thrust::device_vector<uint32_t> indicesSorted(P);
    thrust::sequence(indices.begin(), indices.end());

    // Query temp storage
    cub::DeviceRadixSort::SortPairs(
        nullptr, tempBytes,
        mortonCodes.data().get(), mortonCodesSorted.data().get(),
        indices.data().get(), indicesSorted.data().get(),
        P);
    if (tempBytes > tempStorage.size()) tempStorage.resize(tempBytes);

    // Execute sort
    cub::DeviceRadixSort::SortPairs(
        tempStorage.data().get(), tempBytes,
        mortonCodes.data().get(), mortonCodesSorted.data().get(),
        indices.data().get(), indicesSorted.data().get(),
        P);

    // ----- Step 4: Build boxes (AABBs) -----
    const uint32_t numBoxes = (P + BOX_SIZE - 1) / BOX_SIZE;
    thrust::device_vector<MinMax> boxes(numBoxes);

    boxMinMaxKernel<<<numBoxes, BOX_SIZE>>>(
        P, points,
        indicesSorted.data().get(),
        boxes.data().get());

    // ----- Step 5: Run box-pruned KNN -----
    knn3BoxPrunedKernel<<<numBoxes, BOX_SIZE>>>(
        P, points,
        indicesSorted.data().get(),
        boxes.data().get(),
        numBoxes,
        meanDists);
}