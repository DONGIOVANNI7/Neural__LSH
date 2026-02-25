#include "hypercube.h"
#include "vec.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <queue>
#include <unordered_set>
#include <functional>

HyperCubeSolver::HyperCubeSolver(const Matrix& base, uint32_t seed, uint32_t kproj, uint32_t M, uint32_t probes, float w)
    : ANNSolver(base), kproj(kproj), M(M), probes(probes), w(w), seed(seed), rng(seed) {

    // Professional parameter validation
    if (kproj == 0) throw std::invalid_argument("kproj cannot be zero");
    if (kproj > 64) throw std::invalid_argument("kproj cannot exceed 64");
    if (w <= 0.0f) throw std::invalid_argument("Window width must be positive");
}

void HyperCubeSolver::build() {
    size_t d = base[0].size();

    // Initialize random projections, offsets and bitseeds
    proj.resize(kproj);
    offsets.resize(kproj);
    bit_seeds.resize(kproj);

    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(0.0f, w);

    for (uint32_t i = 0; i < kproj; ++i) {
        proj[i].resize(d);
        for (size_t j = 0; j < d; ++j) {
            proj[i][j] = normal(rng);
        }
        normalize(proj[i]);
        offsets[i] = uniform(rng);
        bit_seeds[i] = rng();
    }

    // Build the hash table
    table.clear();
    for (uint32_t i = 0; i < base.size(); ++i) {
        uint64_t key = vertex_from_vec(base[i]);
        table[key].push_back(i);
    }
}

uint64_t HyperCubeSolver::vertex_from_vec(const Vec& vec) const {
    uint64_t key = 0;
    for (uint32_t i = 0; i < kproj; ++i) {
        float dp = proj[i] * vec;

        // NUMERICAL STABILITY: Avoid division by extremely small w
        float safe_w = std::max(w, std::numeric_limits<float>::epsilon());
        float projected_value = (dp + offsets[i]) / safe_w;

        // NUMERICAL STABILITY: Handle NaN/infinity gracefully
        if (!std::isfinite(projected_value)) {
            projected_value = 0.0f;
        }

        int64_t bin_index = static_cast<int64_t>(std::floor(projected_value));

        // --- Randomized uniform mapping from h to {0,1} ---
        // Combine bucket index with stored random seed for this projection
        uint64_t hash_input = static_cast<uint64_t>(bin_index) ^ static_cast<uint64_t>(bit_seeds[i]);
        
        // Use std::hash to get uniform distribution
        std::hash<uint64_t> hasher;
        uint64_t hashed_value = hasher(hash_input);
        
        // Extract the least significant bit as our uniform random bit
        uint8_t bit = hashed_value & 1ULL;
        
        // Store the bit in the key
        if (bit) {
            key |= (uint64_t(1) << i);
        }
    }
    return key;
}

void HyperCubeSolver::probe_neighbors_priority(uint64_t base_vertex, size_t max_probes, std::vector<uint64_t>& out) const {
    if (max_probes == 0) return;

    struct VertexPriority {
        uint64_t vertex;
        int hamming_distance;

        // Min-heap: lower Hamming distance has higher priority
        bool operator<(const VertexPriority& other) const {
            return hamming_distance > other.hamming_distance;
        }
    };

    std::priority_queue<VertexPriority> pq;
    std::unordered_set<uint64_t> visited;

    // Start with base vertex
    pq.push({base_vertex, 0});
    visited.insert(base_vertex);

    while (!pq.empty() && out.size() < max_probes) {
        VertexPriority current = pq.top();
        pq.pop();
        out.push_back(current.vertex);

        // Generate neighbors by flipping one bit at a time
        for (uint32_t i = 0; i < kproj && out.size() < max_probes; ++i) {
            uint64_t neighbor = current.vertex ^ (uint64_t(1) << i);

            // skip visited neighbors
            if (!visited.insert(neighbor).second) continue;

            pq.push({neighbor, current.hamming_distance + 1});
            out.push_back(neighbor);
        }
    }
}

QueryResult HyperCubeSolver::knn(const Vec& q, uint32_t N) const {
    if (N == 0) return QueryResult();

    // Step 1: Get candidate indices using priority-based neighbor generation
    std::vector<uint32_t> candidates;
    candidates.reserve(M);

    uint64_t base_vertex = vertex_from_vec(q);
    std::vector<uint64_t> vertices_to_probe;
    probe_neighbors_priority(base_vertex, probes, vertices_to_probe);

    for (uint64_t vertex : vertices_to_probe) {
        if (candidates.size() >= M) break;

        auto it = table.find(vertex);
        if (it == table.end()) continue;

        for (uint32_t id : it->second) {
            if (candidates.size() < M) {
                candidates.push_back(id);
            } else {
                break;
            }
        }
    }

    // Step 2: Use max-heap to keep track of top-N smallest distances
    std::priority_queue<ResultPair> max_heap;

    for (uint32_t id : candidates) {
        const Vec& vec = base[id];
        if (vec == q) continue;
        float dist = std::sqrt(l2_sq(vec, q));
        heap_insert(max_heap, ResultPair(dist, id), N);
    }

    return result_from_heap(max_heap);
}

RangeResult HyperCubeSolver::range_search(const Vec& q, float R) const {
    RangeResult result;
    if (R < 0.0f) return result;

    float R_sq = R * R;

    // Get candidate indices
    std::vector<uint32_t> candidates;
    candidates.reserve(M);

    uint64_t base_vertex = vertex_from_vec(q);
    std::vector<uint64_t> vertices_to_probe;
    probe_neighbors_priority(base_vertex, probes, vertices_to_probe);

    for (uint64_t vertex : vertices_to_probe) {
        if (candidates.size() >= M) break;

        auto it = table.find(vertex);
        if (it == table.end()) continue;

        for (uint32_t id : it->second) {
            if (candidates.size() < M) {
                candidates.push_back(id);
            } else {
                break;
            }
        }
    }

    // Check which candidates are within range
    for (uint32_t id : candidates) {
        const Vec& vec = base[id];
        if (vec == q) continue;
        float dist_sq = l2_sq(vec, q);
        if (dist_sq <= R_sq) {
            result.emplace_back(std::sqrt(dist_sq), id);
        }
    }

    return result;
}