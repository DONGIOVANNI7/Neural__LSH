#pragma once
#include <cmath>
#include <vector>
#include <stdexcept>

using Vec = std::vector<float>;

inline float operator*(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) throw std::runtime_error("dot: Incompatible vector sizes");
    float result = 0.0f;
    for (size_t i=0; i<a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

inline Vec operator-(const Vec& lhs, const Vec& rhs) {
    if (lhs.size() != rhs.size()) throw std::runtime_error("minus: Incompatible vector sizes");
    Vec result(lhs.size(), 0);
    for (size_t i=0; i<lhs.size(); ++i) {
        result[i] = lhs[i] - rhs[i];
    }
    return result;
}

inline float l2_sq(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) throw std::runtime_error("l2_sq: Incompatible vector sizes");
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = a[i] - b[i];
        result += d * d;
    }
    return result;
}

inline void normalize(Vec& a) {
    float distance = 0;
    for (auto& e : a) {
        distance += e*e;
    }
    distance = std::sqrt(distance);
    for (auto& e : a) {
        e /= distance;
    }
}
