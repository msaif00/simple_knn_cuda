//
// Created by msp on 2/27/26.
//

#pragma once

#include <vector>
#include <cuda_runtime.h>
//For each point compute the mean squared distance to 3-NN
std::vector<float> knn3_mean_dist2_cpu(const std::vector<float3>& points);
