#pragma once
#include "datasets.h"
#include <cstdint>
#include <stdexcept>

using ResultPair = std::pair<float, uint32_t>; // distance, index
using QueryResult = std::vector<ResultPair>;
using RangeResult = std::vector<ResultPair>;

struct BuildOptions {
    unsigned seed = 1;
};

class ANNSolver {
public:
    ANNSolver(const Matrix& _base): base(_base) {
        if (base.empty()) throw std::invalid_argument("Cannot build solver with empty base");
    }
    virtual ~ANNSolver() = default;
    virtual void build() = 0;
    virtual QueryResult knn(const Vec& q, uint32_t N) const = 0;
    virtual RangeResult range_search(const Vec& q, float R) const = 0;

protected:
    const Matrix& base;
};
