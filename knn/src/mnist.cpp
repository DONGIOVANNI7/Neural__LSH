#include "datasets.h"
#include <cstdint>
#include <fstream>
#include <stdexcept>

static uint32_t be32(const unsigned char* p){
    return (uint32_t)p[0]<<24 | (uint32_t)p[1]<<16 | (uint32_t)p[2]<<8 | (uint32_t)p[3];
}

Matrix load_idx3_ubyte(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open mnist file: " + path);

    // Read MNIST header
    unsigned char header[16];
    f.read(reinterpret_cast<char *>(header), sizeof(header));
    if (!f) throw std::runtime_error("Bad mnist file: " + path);

    uint32_t magic_num = be32(header);
    uint32_t nimg = be32(header+4);
    uint32_t nrows = be32(header+8);
    uint32_t ncols = be32(header+12);

    if (magic_num != 2051)
        throw std::runtime_error("Bad MNIST magic number (expected 2051): " + std::to_string(magic_num));

    // Read MNIST data
    Matrix m;
    m.reserve(nimg);
    size_t D = static_cast<size_t>(nrows) * static_cast<size_t>(ncols);
    std::vector<unsigned char> buffer(D);
    for (uint32_t i=0; i<nimg; ++i) {
        f.read(reinterpret_cast<char*>(buffer.data()), D);
        Vec v(D);
        for (size_t j=0; j<D; ++j) v[j] = static_cast<float>(buffer[j]);
        m.emplace_back(std::move(v));
    }
    return m;
}

Dataset load_mnist(const std::string& data_file, const std::string& query_file) {
    Dataset result;
    result.data = load_idx3_ubyte(data_file);
    result.query = load_idx3_ubyte(query_file);
    return result;
}

Dataset load_mnist(const std::string& data_file) {
    Dataset result;
    result.data = load_idx3_ubyte(data_file);
    return result;
}
