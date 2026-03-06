#pragma once

#include <cstdint>
#include <cfloat>
#include <cuda_runtime.h>

#define BOX_SIZE 256
// number of sorted points per box, must be a power of 2 for
// shared memory reduction.

struct MinMax {
    float3 minn;
    float3 maxx;
};

__device__ __host__ inline float distBoxPoint(const MinMax& box, const float3& p) {
    float dx = 0.0f, dy= 0.0f, dz=0.0f;
    if (p.x < box.minn.x) dx = box.minn.x - p.x;
    else if (p.x > box.minn.x) dx=p.x - box.minn.x;

    if (p.y < box.minn.y) dy = box.minn.y -  p.y;
    else if (p.y > box.maxx.y) dy = p.y - box.maxx.y;

    if (p.z < box.minn.z) dz = box.minn.z - p.z;
    else if (p.z > box.maxx.z) dz = p.z - box.maxx.z;

    return dx*dx+dy*dy+dz*dz;
}

//Kernel 1: Compute the AABB of each box

__global__ void boxMinMaxKernel(
    int n,
    const float3* __restrict__ points,
    const uint32_t* __restrict__ sortedIndices,
    MinMax* __restrict__ boxes) {
    const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    //Each thread loads one point
    MinMax me;
    if (globalIdx < n) {
        float3 p = points[sortedIndices[globalIdx]];
        me.minn = p;
        me.maxx = p;
    } else {
        me.minn = make_float3( FLT_MAX, FLT_MAX, FLT_MAX);
        me.maxx = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }

    //Shared memory for reduction
    __shared__ MinMax smem[BOX_SIZE];

    // Classic tree reduction, halve the active threads each step
    for (int offset = BOX_SIZE / 2; offset>=1; offset /= 2) {
        if (threadIdx.x < 2 * offset)
            smem[threadIdx.x] = me;
        __syncthreads();

        //Lower half combines with upper half
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

// Kernel 2: box-pruned exact KNN-3
//
// for each point, loop over all the boxes, skip boxes whose AABB distance
// is already worse than the third best option in the best[] array. Otherwise
// scan the box's points
// Launch config: <<<(N+255)/256, 256>>> (one thread per point in sorted order.

__global__ void knn3BoxPrunedKernl(
    int n,
    const float3* __restrict__ points,
    const uint32_t* __restrict__ sortedIndices,
    const MinMax* __restrict__ boxes,
    int numBoxes,
    float* __restrict__ outMeanDist2) {

    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;

    const uint32_t i = sortedIndices[s];
    const float3 pi = points[i];

    float best[3] = {FLT_MAX, FLT_MAX, FLT_MAX};

    //Pass 1: Check small local window first
    {
        const int lo = max(0, s - 8);
        const int hi = min(n - 1, s + 8);
        for (int t = lo; t <= hi; t++) {
            if (t == s) continue;
            const uint32_t j = sortedIndices[t];
            const float3 pj = points[j];
            const float dx = pi.x - pj.x;
            const float dy = pi.y - pj.y;
            const float dz = pi.z - pj.z;
            float dist2 = dx*dx + dy*dy + dz*dz;
            // Bubble-insert into best[3]
            for (int k = 0; k < 3; k++) {
                if (dist2 < best[k]) {
                    float tmp = best[k];
                    best[k] = dist2;
                    dist2 = tmp;
                }
            }
        }
    }

    //reject ->current worst of the 3-best
    float reject = best[2];

    //Reset for all full extract search

    best[0] = FLT_MAX;
    best[1] = FLT_MAX;
    best[2] = FLT_MAX;

    //Pass 2: scan all boxes
    for (int b = 0; b < numBoxes; b++) {
        // If the box's closest possible point is farther than our reject threshold,
        // AND farther than our current 3rd-best, skip the entire box.
        float boxDist = distBoxPoint(boxes[b], pi);
        if (boxDist > reject || boxDist > best[2])
            continue;

        // Scan points in this box
        const int boxStart = b * BOX_SIZE;
        const int boxEnd   = min(n, (b + 1) * BOX_SIZE);
        for (int t = boxStart; t < boxEnd; t++) {
            if (t == s) continue;
            const uint32_t j = sortedIndices[t];
            const float3 pj = points[j];
            const float dx = pi.x - pj.x;
            const float dy = pi.y - pj.y;
            const float dz = pi.z - pj.z;
            float dist2 = dx*dx + dy*dy + dz*dz;
            for (int k = 0; k < 3; k++) {
                if (dist2 < best[k]) {
                    float tmp = best[k];
                    best[k] = dist2;
                    dist2 = tmp;
                }
            }
        }
    }

    outMeanDist2[i] = (best[0] + best[1] + best[2]) / 3.0f;
}











