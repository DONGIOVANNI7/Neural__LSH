#include "cli.h"
#include "datasets.h"
#include "hypercube.h"
#include "ivfflat.h"
#include "ivfpq.h"
#include "lsh.h"
#include "search_interface.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <execution>

static CLI cli;
static std::mutex outMutex;

// Run search for single query and return whether true nearest neighbor is found in N approximate neighbors
void search(
    const Vec& query,
    const size_t idx,
    const std::unique_ptr<ANNSolver>& approxSolver
) {
    QueryResult approxQueryResult = approxSolver->knn(query, cli.N);

    uint32_t N = std::min(static_cast<uint32_t>(approxQueryResult.size()), cli.N);

    std::lock_guard<std::mutex> lock(outMutex);
    std::cout << idx << std::endl;
    for (uint32_t n=0; n<N; ++n) {
        size_t indexApproximate = approxQueryResult[n].second;
        std::cout << indexApproximate << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse cli arguments
    if (!cli.parse(argc, argv)) return -1;

    // Load dataset
    Dataset dataset;
    switch (cli.type) {
        case MNIST:
            dataset = load_mnist(cli.dfile, cli.dfile);
            break;
        case SIFT:
            dataset = load_sift(cli.dfile, cli.dfile);
            break;
    }

    // Setup output
    std::unique_ptr<std::ostream> ofsp;

    // Initialize solver
    std::unique_ptr<ANNSolver> approxSolver;
    switch (cli.method) {
        case LSH:
            approxSolver = std::make_unique<LSHSolver>(dataset.data, cli.seed, cli.L, cli.k, cli.w);
            break;
        case HYPERCUBE:
            approxSolver = std::make_unique<HyperCubeSolver>(dataset.data, cli.seed, cli.kproj, cli.M, cli.probes, cli.w);
            break;
        case IVFFLAT:
            approxSolver = std::make_unique<IVFFlatSolver>(dataset.data, cli.seed, cli.kclusters, cli.nprobe);
            break;
        case IVFPQ:
            approxSolver = std::make_unique<IVFPQSolver>(dataset.data, cli.seed, cli.kclusters, cli.nprobe, cli.M, cli.nbits);
            break;
        default:
            break;
    }
    approxSolver->build();

    std::for_each(
        std::execution::par,
        dataset.query.begin(),
        dataset.query.end(),
        [&] (Vec& query) {
            size_t idx = &query - &dataset.query[0];
            search(query, idx, approxSolver);
        }
    );

    return 0;
}
