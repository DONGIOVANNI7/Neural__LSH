#pragma once

#include "cli.h"
#include "kmeans.h"
#include "search_interface.h"
#include <cstdint>


class IVFPQSolver : public ANNSolver {
public:
    IVFPQSolver(const Matrix& _base, uint32_t seed, uint32_t _kclusters, uint32_t _nprobe, uint32_t M, uint32_t nbits);
    void build() override;
    QueryResult knn(const Vec& q, uint32_t N) const override;
    RangeResult range_search(const Vec& q, float R) const override;
    double getSilhouette() {return silhouette;}

private:
    uint32_t seed = 1, kclusters = 50, nprobe = 5;
    uint32_t M = 16, nbits = 8;

    uint32_t s = 256;

    double silhouette = 0;

    // IVF
    std::vector<Vec> centroids;
    using ListItem = std::pair<uint32_t, std::vector<size_t>>;
    std::vector<std::vector<ListItem>> lists;
    std::vector<double> clusterRadius;

    // PQ
    std::vector<size_t> lengths; // lengths of M vector parts
    std::vector<size_t> offsets; // offsets of M vector parts
    std::vector<std::vector<Vec>> codebooks; // local centroids for M spaces

    std::vector<std::vector<double>> computeLUT(const Vec& q, const size_t cIdx) const;
    double ASD(const std::vector<std::vector<double>>& lut, const ListItem& listItem) const;
};
