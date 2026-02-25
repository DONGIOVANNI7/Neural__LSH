#pragma once
#include "vec.h"
#include <string>

enum DatasetType {
    MNIST,
    SIFT
};

using Matrix = std::vector<Vec>;

struct Dataset {
    Matrix data;
    Matrix query;
};


// Load sift dataset and query
Dataset load_sift(const std::string& datafile, const std::string& queryfile);
// Load sift dataset only
Dataset load_sift(const std::string& datafile);
// Load mnist dataset and query
Dataset load_mnist(const std::string& datafile, const std::string& queryfile);
// Load mnist dataset only
Dataset load_mnist(const std::string& datafile);
