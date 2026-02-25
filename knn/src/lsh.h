#pragma once
#include "search_interface.h"
#include <cstdint>
#include <unordered_map>
#include <vector>

class LSHSolver : public ANNSolver {
public:
    LSHSolver(const Matrix& _base, uint32_t seed, uint32_t _L, uint32_t _k, float _w);
    void build() override;
    QueryResult knn(const Vec& q, uint32_t N) const override;
    RangeResult range_search(const Vec& q, float R) const override;

private:
    // LSH params
    uint32_t seed = 1, nHashTables = 5, nHashFuncs = 4;
    uint32_t tableSize;
    float cellSize = 4.0f;

    static constexpr uint32_t PRIME = (1UL << 32) - 5; // big prime for universal hashing

    // for each hash table i and hash function j
    std::vector<std::vector<Vec>> V; // projection vectors
    std::vector<std::vector<float>> T; // offsets
    std::vector<std::vector<uint32_t>> R; // combination factors

    std::vector<std::unordered_map<uint64_t, std::vector<uint32_t>>> hashTables;

    uint64_t g(int tableIndex, const Vec& vector) const;
    std::vector<int32_t> compute_h(int tableIndex, const Vec& vector) const;
};
