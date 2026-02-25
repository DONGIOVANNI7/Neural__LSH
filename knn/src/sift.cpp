#include "datasets.h"
#include <fstream>
#include <stdexcept>

Matrix load_fvecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open fvecs: " + path);

    Matrix m;
    while (true) {
        int32_t d;
        f.read(reinterpret_cast<char *>(&d), sizeof(d));

        if (f.eof()) break;

        std::vector<float> v;
        v.resize(static_cast<size_t>(d));

        if (!f.read(reinterpret_cast<char *>(v.data()), sizeof(float)*static_cast<size_t>(d))) {
            throw std::runtime_error("Misformed vector");
        };

        m.emplace_back(std::move(v));
    }
    return m;
}

Dataset load_sift(const std::string& datafile, const std::string& queryfile) {
    Dataset result;
    result.data = load_fvecs(datafile);
    result.query = load_fvecs(queryfile);
    return result;
}

Dataset load_sift(const std::string& datafile) {
    Dataset result;
    result.data = load_fvecs(datafile);
    return result;
}
