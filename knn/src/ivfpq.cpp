#include "ivfpq.h"
#include "kmeans.h"
#include "search_interface.h"
#include "utils.h"
#include "vec.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <unordered_set>

IVFPQSolver::IVFPQSolver(
    const Matrix& _base,
    uint32_t _seed,
    uint32_t _kclusters,
    uint32_t _nprobe,
    uint32_t _M,
    uint32_t _nbits
) : ANNSolver(_base), seed(_seed), kclusters(_kclusters), nprobe(_nprobe), M(_M), nbits(_nbits) {
    if (kclusters == 0) throw std::invalid_argument("kclusters cannot be zero");
    if (nprobe == 0) throw std::invalid_argument("nprobe cannot be zero");
    if (M == 0) throw std::invalid_argument("M cannot be zero");
    if (nbits == 0) throw std::invalid_argument("nbits cannot be zero");

    s = 1 << nbits;
    lengths.resize(M);
    offsets.resize(M);
    codebooks.reserve(M);
}

void IVFPQSolver::build() {
    KMeansResult km = kmeans_lloyd(base, kclusters, seed, 10);
    silhouette = silhouette_approx(base, km);

    centroids = std::move(km.centroids);
    clusterRadius = std::move(km.clusterRadius);
    lists.assign(centroids.size(), {});

    // Calculate residual vectors
    std::vector<Vec> residuals;
    residuals.reserve(km.labels.size());
    for (size_t i=0; i<km.labels.size(); ++i) {
        uint32_t cId = km.labels[i];
        residuals.emplace_back(std::move(base[i] - centroids[cId]));
    }

    // Define PQ subspaces and train PQ codebook
    size_t dim = base[0].size();
    size_t base_len = dim / M;
    size_t rem = dim % M;
    size_t offset = 0;
    for (uint32_t m=0; m<M; ++m) {
        size_t len = base_len + (m < rem ? 1 : 0);
        lengths[m] = len;
        offsets[m] = offset;

        std::vector<Vec> Xm;
        Xm.reserve(base.size());
        for (size_t idx=0; idx<base.size(); ++idx) {
            const Vec& residual = residuals[idx];
            Xm.emplace_back(residual.begin()+offset, residual.begin()+offset+len);
        }
        offset += len;

        KMeansResult kmM = kmeans_lloyd(Xm, s, seed + m, 5);
        codebooks.emplace_back(std::move(kmM.centroids));
    }

    // Encode vectors with PQ code
    for (size_t idx=0; idx<residuals.size(); ++idx) {
        uint32_t cId = km.labels[idx];
        std::vector<size_t> codesM(M);
        for (uint32_t m=0; m<M; ++m) {
            size_t off = offsets[m];
            size_t len = lengths[m];
            Vec Xm(residuals[idx].begin()+off, residuals[idx].begin()+off+len);

            auto result = std::min_element(codebooks[m].begin(), codebooks[m].end(), [&](const Vec& a, const Vec& b){return l2_sq(a, Xm) < l2_sq(b, Xm);});
            codesM[m] = std::distance(codebooks[m].begin(), result);
        }
        lists[cId].emplace_back(idx, std::move(codesM));
    }
}

// Encode query vector
std::vector<std::vector<double>> IVFPQSolver::computeLUT(const Vec& q, const size_t cIdx) const {
    std::vector<std::vector<double>> lut(M);

    const Vec& c = centroids[cIdx];
    Vec residual = q - c;
    for (uint32_t m=0; m<M; ++m) {
        lut[m].resize(s);
        size_t off = offsets[m];
        size_t len = lengths[m];
        Vec Rm(residual.begin()+off, residual.begin()+off+len);

        for (size_t h=0; h<s; ++h) {
            auto& code = codebooks[m][h];
            lut[m][h] = l2_sq(Rm, code);
        }
    }

    return lut;
}

double IVFPQSolver::ASD(const std::vector<std::vector<double>>& lut, const ListItem& listItem) const {
    double distance = 0;
    for (size_t i=0; i<lut.size(); ++i) {
        distance += lut[i][listItem.second[i]];
    }
    return std::sqrt(distance);
}

QueryResult IVFPQSolver::knn(const Vec& q, uint32_t N) const {
    if (N == 0) return QueryResult();
    std::priority_queue<ResultPair> heap;
    std::unordered_set<uint32_t> seen;

    for (auto& cIdx : nearestClusters(centroids, q, nprobe)) {
        auto lut = computeLUT(q, cIdx);

        for (const ListItem& listItem : lists[cIdx]) {
            if (!seen.insert(listItem.first).second) continue;
            double distance = ASD(lut, listItem);
            ResultPair element = ResultPair(distance, listItem.first);
            heap_insert(heap, element, N);
        }
    }

    return result_from_heap(heap);
}


RangeResult IVFPQSolver::range_search(const Vec& q, float R) const {
    RangeResult result;
    std::unordered_set<uint32_t> seen;

    for (auto& cIdx : nearestClusters(centroids, q, nprobe)) {
        auto lut = computeLUT(q, cIdx);

        for (const ListItem& listItem : lists[cIdx]) {
            if (!seen.insert(listItem.first).second) continue;
            double distance = ASD(lut, listItem);
            if (distance < R) result.emplace_back(distance, listItem.first);
        }
    }
    return result;
}
