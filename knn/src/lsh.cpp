#include "lsh.h"
#include "search_interface.h"
#include "utils.h"
#include "vec.h"
#include <cmath>
#include <cstdint>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>

LSHSolver::LSHSolver(
    const Matrix& _base,
    uint32_t _seed,
    uint32_t _L,
    uint32_t _k,
    float _w
) : ANNSolver(_base), seed(_seed), nHashTables(_L), nHashFuncs(_k) {

    if (nHashFuncs == 0) throw std::invalid_argument("k cannot be zero");
    if (nHashTables == 0) throw std::invalid_argument("L cannot be zero");
    if (_w <= 0) throw std::invalid_argument("w must be positive");
    cellSize = _w;
    tableSize = static_cast<uint32_t>(base.size() / 4);
}

void LSHSolver::build() {
    if (base.empty()) return;
    const size_t dim = base[0].size();

    V.assign(nHashTables, std::vector<Vec>(nHashFuncs, Vec(dim, 0.0f)));
    T.assign(nHashTables, std::vector<float>(nHashFuncs, 0.0f));
    R.assign(nHashTables, std::vector<uint32_t>(nHashFuncs, 1));
    hashTables.clear();
    hashTables.resize(nHashTables);

    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> unifT(0.0f, cellSize);
    std::uniform_int_distribution<uint32_t> unifA(1, PRIME - 1);

    for (uint32_t tbl = 0; tbl < nHashTables; ++tbl) {
        for (uint32_t i = 0; i < nHashFuncs; ++i) {
            for (size_t d = 0; d < dim; ++d)
                V[tbl][i][d] = gauss(rng);
            normalize(V[tbl][i]);

            T[tbl][i] = unifT(rng);
            // R[tbl][i] = unifA(rng); // when R is random there are very few collisions
        }
    }

    for (uint32_t tbl = 0; tbl < nHashTables; ++tbl) {
        auto& table = hashTables[tbl];
        for (size_t idx = 0; idx < base.size(); ++idx) {
            const Vec& vector = base[idx];
            uint64_t key = g(tbl, vector);
            table[key].push_back(static_cast<uint32_t>(idx));
        }
    }
}

std::vector<int32_t> LSHSolver::compute_h(int tableIndex, const Vec& vector) const {
    std::vector<int32_t> h(nHashFuncs);
    for (uint32_t i = 0; i < nHashFuncs; ++i) {
        float projection = (vector * V[tableIndex][i]) + T[tableIndex][i];
        h[i] = static_cast<int32_t>(std::floor(projection / cellSize));
    }
    return h;
}

uint64_t LSHSolver::g(int tableIndex, const Vec& vector) const {
    const auto h = compute_h(tableIndex, vector);

    // g = Σ (r_i * (h_i mod p)) mod p, with safe 64-bit modmul and positive remainder
    uint64_t key = 0;
    for (uint32_t i = 0; i < nHashFuncs; ++i) {
        int64_t hi = static_cast<int64_t>(h[i]) % static_cast<int64_t>(PRIME);
        if (hi < 0) hi += PRIME;
        uint64_t term = static_cast<uint64_t>(R[tableIndex][i]) * static_cast<uint64_t>(hi);
        key += term % PRIME;
        key %= PRIME;
    }
    return key % tableSize;
}

QueryResult LSHSolver::knn(const Vec& q, uint32_t N) const {
    if (N == 0) return QueryResult();

    std::priority_queue<ResultPair> heap;
    std::unordered_set<uint32_t> seen;
    seen.reserve(1024 * nHashTables);

    for (uint32_t tbl = 0; tbl < nHashTables; ++tbl) {
        uint64_t key = g(tbl, q);
        const auto it = hashTables[tbl].find(key);
        if (it == hashTables[tbl].end()) {
            continue;
        }

        const auto& indices = it->second;
        for (auto idx : indices) {
            if (!seen.insert(idx).second) continue; // avoid duplicates
            const Vec& vec = base[idx];
            if (vec == q) continue;
            float distance = std::sqrt(l2_sq(q, vec));
            heap_insert(heap, ResultPair(distance, idx), N);
        }
    }

    return result_from_heap(heap);
}

RangeResult LSHSolver::range_search(const Vec& q, float R) const {
    RangeResult result;
    const float R_sq = R * R;
    std::unordered_set<uint32_t> seen;

    for (uint32_t tbl = 0; tbl < nHashTables; ++tbl) {
        uint64_t key = g(tbl, q);
        const auto it = hashTables[tbl].find(key);
        if (it == hashTables[tbl].end()) continue;

        const auto& indices = it->second;
        for (auto idx : indices) {
            if (!seen.insert(idx).second) continue; // avoid duplicates
            const Vec& vec = base[idx];
            if (vec == q) continue;
            float dist_sq = l2_sq(q, vec);
            if (dist_sq < R_sq) result.emplace_back(std::sqrt(dist_sq), idx);
        }
    }
    return result;
}
