#pragma once

#include "search_interface.h"
#include <unordered_map>
#include <vector>
#include <random>

class HyperCubeSolver : public ANNSolver {
public:
    HyperCubeSolver(const Matrix& base, uint32_t seed, uint32_t kproj, uint32_t M, uint32_t probes, float w);
    void build() override;
    QueryResult knn(const Vec& q, uint32_t N) const override;
    RangeResult range_search(const Vec& q, float R) const override;

private:
    uint32_t kproj;
    uint32_t M;
    uint32_t probes;
    float w;
    uint32_t seed;

    std::mt19937 rng;
    std::vector<Vec> proj;  // kproj projection vectors, each of dimension d
    std::vector<float> offsets; // kproj offsets
    std::vector<uint64_t> bit_seeds; // Random seeds for uniform bit mapping
    std::unordered_map<uint64_t, std::vector<uint32_t>> table;

    uint64_t vertex_from_vec(const Vec& vec) const;
    void probe_neighbors_priority(uint64_t base, size_t max_probes, std::vector<uint64_t>& out) const;
};
