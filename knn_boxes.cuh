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
    else if (p.x > box.minn.x) dx=p.x - boxx.minn.x;

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
    const int globalIdx = blockIdx.x * blockDim.x = threadIdx.x;

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












