#include "ivfflat.h"
#include "kmeans.h"
#include "utils.h"
#include "vec.h"
#include <cmath>
#include <queue>
#include <stdexcept>
#include <unordered_set>

IVFFlatSolver::IVFFlatSolver(
    const Matrix& _base,
    uint32_t _seed,
    uint32_t _kclusters,
    uint32_t _nprobe
) : ANNSolver(_base), seed(_seed), kclusters(_kclusters), nprobe(_nprobe) {
    if (kclusters == 0) throw std::invalid_argument("kclusters cannot be zero");
    if (nprobe == 0) throw std::invalid_argument("nprobe cannot be zero");
}

void IVFFlatSolver::build() {
    KMeansResult km = kmeans_lloyd(base, kclusters, seed, 10);
    silhouette = silhouette_approx(base, km);

    centroids = std::move(km.centroids);
    clusterRadius = std::move(km.clusterRadius);

    lists.assign(centroids.size(), {});
    for (size_t i=0; i<km.labels.size(); ++i) {
        uint32_t label = km.labels[i];
        lists[label].push_back(static_cast<uint32_t>(i));
    }
}

QueryResult IVFFlatSolver::knn(const Vec& q, uint32_t N) const {
    if (N == 0) return QueryResult();

    std::unordered_set<uint32_t> seen;
    std::priority_queue<ResultPair> heap;

    for (auto& cIdx : nearestClusters(centroids, q, nprobe)) {
        for (uint32_t idx : lists[cIdx]) {
            if (!seen.insert(idx).second) continue;
            const Vec& vec = base[idx];
            if (vec == q) continue;
            float distance = std::sqrt(l2_sq(vec, q));
            ResultPair element = ResultPair(distance, idx);
            heap_insert(heap, element, N);
        }
    }

    return result_from_heap(heap);
}
RangeResult IVFFlatSolver::range_search(const Vec& q, float R) const {
    RangeResult result;
    float R_sq = R*R;
    std::unordered_set<uint32_t> seen;

    for (auto& cIdx : nearestClusters(centroids, q, nprobe)) {
        for (uint32_t idx : lists[cIdx]) {
            if (!seen.insert(idx).second) continue;
            const Vec& vec = base[idx];
            if (vec == q) continue;
            float dist_sq = l2_sq(vec, q);
            if (dist_sq < R_sq) result.emplace_back(std::sqrt(dist_sq), idx);
        }
    }
    return result;
}
