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

