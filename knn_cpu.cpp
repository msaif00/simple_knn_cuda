//
// Created by msp on 2/27/26.
//

#include "knn_cpu.h"

#include <cfloat>
#include <algorithm>

//Method for updating best[3]
static inline void updateKBest3(float dist2, float best[3]):
{
    for (int j=0; j < 3; j++) {
        if (dist2 < best[j]) {
            float t = best[j];
            best[j] = dist2;
            dist2 = t;
        }
    }
}

std::vector<float> knn3_mean_dist2_cpu(const std::vector<float3>& points) {
    const int n =static_cast<int>(points.size());
    std::vector<float> out(n, 0.0f);

    for (int i=0; i < n; i++) {
        const float3 pi = points[i];
        float best[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
        for (int j=0; j < n; j++) {
            if (j==i ) continue;
            const float3 pj =points[j];
            const float dx = pi.x - pj.x;
            const float dy = pi.y - pj.y;
            const float dz = pi.z - pj.z;
            const float dist2 = dx*dx + dy*dy + dz*dz;
            updateKBest3(dist2, best);
        }
        out[i] = (best[0] + best[1] + best[2]) / 3.0f;
    }
    return out;
}