#include "exact.h"
#include "search_interface.h"
#include "utils.h"
#include "vec.h"
#include <cmath>
#include <queue>

QueryResult ExactSolver::knn(const Vec& q, uint32_t N) const {
    if (N == 0) return QueryResult();

    std::priority_queue<ResultPair> heap;

    for (size_t idx=0; idx<base.size(); ++idx) {
        const Vec& vec = base[idx];
        if (vec == q) continue;
        float distance = std::sqrt(l2_sq(vec, q));
        ResultPair element = ResultPair(distance, idx);
        heap_insert(heap, element, N);
    }

    return result_from_heap(heap);
}

RangeResult ExactSolver::range_search(const Vec& q, float R) const {
    RangeResult result;

    float R_sq = R*R;
    for (size_t idx=0; idx<base.size(); ++idx) {
        const Vec& vec = base[idx];
        if (vec == q) continue;
        float distance_sq = l2_sq(q, vec);
        if (distance_sq < R_sq) result.push_back(ResultPair(std::sqrt(distance_sq), idx));
    }

    return result;
}
