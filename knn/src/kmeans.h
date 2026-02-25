#pragma once

#include "datasets.h"
#include <cstdint>
#include <vector>

struct KMeansResult {
    std::vector<Vec> centroids;
    std::vector<uint32_t> labels; // labels[i] = cluster index of vector i
    std::vector<double> clusterRadius;
};

// Run kmeans on a Matrix in order to find k clusters
KMeansResult kmeans_lloyd(const Matrix& X, uint32_t k, uint32_t seed, uint32_t maxIter);
std::vector<uint32_t> nearestClusters(const std::vector<Vec>& centroids, const Vec& q, uint32_t nprobe);

// Fast approximate silhouette using centroid distances and cluster mean radii (no O(n^2)).
// For point i in cluster A:
//   a_i ~ mean distance to its own centroid (use clusterRadius[A]).
//   b_i ~ min over clusters B != A of (distance(centroid_A, centroid_B) + clusterRadius[B]).
// silhouette_i = (b_i - a_i) / max(a_i, b_i).
// Returns average over all points.
double silhouette_approx(const Matrix& X, const KMeansResult& km);
