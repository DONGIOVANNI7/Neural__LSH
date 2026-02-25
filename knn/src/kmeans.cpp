#include "kmeans.h"
#include <algorithm>
#include <cstdint>
#include <queue>
#include <random>

KMeansResult kmeans_lloyd(const Matrix &X, uint32_t k, uint32_t seed, uint32_t maxIter) {
    const size_t n = X.size();
    const size_t d = n ? X[0].size() : 0;
    KMeansResult result;
    if (n==0 || d==0) return result;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> uni(0, n-1);

    // Init: k random points from X as centroids (Forgy)
    result.centroids.assign(k, Vec(d, 0.0f));
    for (uint32_t c=0; c<k; ++c) result.centroids[c] = X[uni(rng)];

    std::vector<uint32_t> labels(n, -1);
    for (uint32_t it=0; it<maxIter; ++it) {
        bool changed = false;
        // Assign
        for (size_t i=0; i<n; ++i) {
            auto it = std::min_element(result.centroids.begin(), result.centroids.end(), [&](const Vec& a, const Vec& b){return l2_sq(a, X[i]) < l2_sq(b, X[i]);});
            size_t bestj = std::distance(result.centroids.begin(), it);
            if (labels[i] != bestj) {
                labels[i] = static_cast<uint32_t>(bestj);
                changed = true;
            }
        }
        // Update
        std::vector<Vec> newc(k, Vec(d, 0.f));
        std::vector<size_t> counts(k, 0);
        for (size_t i=0; i<n; ++i) {
            uint32_t j = labels[i];
            counts[j]++;
            for (size_t t=0; t<d; ++t) newc[j][t] += X[i][t];
        }
        for (uint32_t j=0; j<k; ++j) {
            if (counts[j]==0) {
                newc[j] = X[uni(rng)]; // re-seed empty cluster
            } else {
                for (size_t t=0; t<d; ++t) newc[j][t] /= (float)counts[j];
            }
        }
        result.centroids.swap(newc);
        if (!changed) break;
    }
    result.labels = labels;

    // cluster radii: mean sqrt(dist^2) to centroid per cluster (used for silhouette approx)
    std::vector<double> dist_sum(k, 0.0);
    std::vector<size_t> count(k, 0);
    for (size_t i=0;i<n;++i) {
        uint32_t cIdx = labels[i];
        double dist_sq = l2_sq(X[i], result.centroids[cIdx]);
        dist_sum[cIdx] += std::sqrt(std::max(0.0, dist_sq));
        count[cIdx]++;
    }
    result.clusterRadius.assign(k, 0.0);
    for (uint32_t j=0; j<k; ++j) {
        result.clusterRadius[j] = count[j] ? (dist_sum[j] / (double)count[j]) : 0.0;
    }

    return result;
}

std::vector<uint32_t> nearestClusters(const std::vector<Vec>& centroids, const Vec& q, uint32_t nprobe) {
    using data = std::pair<double, uint32_t>;
    using container = std::vector<data>;
    using comp = std::greater<data>;
    std::priority_queue<data, container, comp> minHeap;

    for (size_t idx=0; idx<centroids.size(); ++idx) {
        double dist = std::sqrt(l2_sq(q, centroids[idx]));
        data clusterDistance = std::make_pair(dist, idx);
        if (minHeap.size() < nprobe) {
            minHeap.push(clusterDistance);
        }
        else {
            if (clusterDistance < minHeap.top()) {
                minHeap.pop();
                minHeap.push(clusterDistance);
            }
        }
    }

    std::vector<uint32_t> result;
    result.reserve(nprobe);
    while (!minHeap.empty()) {
        result.push_back(minHeap.top().second);
        minHeap.pop();
    }
    return result;
}

double silhouette_approx(const Matrix& X, const KMeansResult& km) {
    const size_t n = X.size();
    const size_t k = km.centroids.size();
    if (n==0 || k<=1) return 0.0;

    // precompute centroid-centroid distances
    std::vector<double> cdist(k*k, 0.0);
    for (size_t i=0; i<k; ++i) {
        for (size_t j=i+1; j<k; ++j) {
            double d = std::sqrt(l2_sq(km.centroids[i], km.centroids[j]));
            cdist[i*k+j] = cdist[j*k+i] = d;
        }
    }

    double ssum = 0.0;
    for (size_t i=0; i<n; ++i) {
        uint32_t A = km.labels[i];
        double a = km.clusterRadius[A];
        double b = std::numeric_limits<double>::infinity();
        for (size_t B=0; B<k; ++B) {
            if (B == A) continue;
            double cand = cdist[A*k+B] + km.clusterRadius[B];
            b = std::min(b, cand);
        }
        double s = (b - a) / std::max(a, b);
        ssum += s;
    }
    return ssum / static_cast<double>(n);
}
