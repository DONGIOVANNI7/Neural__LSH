#pragma once

#include "cli.h"
#include "kmeans.h"
#include "search_interface.h"
#include <cstdint>

class IVFFlatSolver : public ANNSolver {
public:
    IVFFlatSolver(const Matrix& _base, uint32_t seed, uint32_t _kclusters, uint32_t _nprobe);
    void build() override;
    QueryResult knn(const Vec& q, uint32_t N) const override;
    RangeResult range_search(const Vec& q, float R) const override;
    double getSilhouette() {return silhouette;}

protected:
    uint32_t seed = 1, kclusters = 50, nprobe = 5;
    double silhouette = 0;
    std::vector<Vec> centroids;
    std::vector<std::vector<uint32_t>> lists;
    std::vector<double> clusterRadius;
};
