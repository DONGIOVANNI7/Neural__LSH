#pragma once

#include "search_interface.h"
#include <algorithm>
#include <chrono>
#include <queue>

// Insert a ResultPair into the max heap QueryResult and keep only N elements
inline void heap_insert(std::priority_queue<ResultPair>& heap, const ResultPair& element, size_t N) {
    // if heap has less than N elements insert element
    // else if element is less than largest element of heap, remove top of heap and insert element
    if (heap.size() < N) {
        heap.emplace(std::move(element));
    } else if (element < heap.top()) {
        heap.pop();
        heap.emplace(std::move(element));
    }
}

inline QueryResult result_from_heap(std::priority_queue<ResultPair>& heap) {
    QueryResult result;
    while (!heap.empty()) {
        // Place largest element from queue into result
        result.emplace_back(std::move(heap.top()));
        heap.pop();
    }
    // Reverse result since it's sorted from largest to smallest element
    // and we want smallest to largest
    std::reverse(result.begin(), result.end());

    return result;
}

struct Timer {
    std::chrono::steady_clock::time_point t0;
    void tic() { t0 = std::chrono::steady_clock::now(); }
    double toc_ms() const {
        auto t1 = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> duration = t1 - t0;
        return duration.count();
    }
};
