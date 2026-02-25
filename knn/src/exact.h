#pragma once

#include "datasets.h"
#include "search_interface.h"

class ExactSolver : public ANNSolver {
public:
    ExactSolver(const Matrix& _base): ANNSolver(_base) {}
    void build() override {}
    QueryResult knn(const Vec& q, uint32_t N) const override;
    RangeResult range_search(const Vec& q, float R) const override;
};
