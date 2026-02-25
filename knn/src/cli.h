#pragma once
#include <cstdint>
#include <string>
#include "datasets.h"

enum SearchMethod {
    NONE,
    LSH,
    HYPERCUBE,
    IVFFLAT,
    IVFPQ
};

struct CLI {
    static constexpr uint32_t kclusters_mnist = 40;
    static constexpr uint32_t kclusters_sift = 2;

    // Global
    DatasetType type = MNIST; // mnist | sift
    std::string dfile;
    uint32_t N = 1;
    uint32_t seed = 1;

    // Method switches
    SearchMethod method = LSH;

    // LSH
    uint32_t k = 4, L = 5;
    float w = 4.0;

    // Hypercube
    uint32_t kproj = 14, M = 10, probes = 2;

    // IVFFlat
    uint32_t kclusters = 40, nprobe = 5;

    // IVFPQ
    uint32_t nbits = 8;

    bool parse(int argc, char** argv);
    void print_help() const;
};
