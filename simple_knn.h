#pragma once
#include <cuda_runtime.h>
class SimpleKNN {
public:
    // Computes the mean squared distance to 3 nearest neighbors for each point.
    //
    // Parameters:
    //   P          - number of points
    //   points     - device pointer to P float3 values
    //   meanDists  - device pointer to P floats (output)
    //
    // All memory must be allocated on the GPU before calling.
    static void knn(int P, float3* points, float* meanDists);
};