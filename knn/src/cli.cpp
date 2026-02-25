#include "cli.h"
#include <iostream>

void CLI::print_help() const {
    std::cout << "Usage: ann_search [global options] [method options]\n\n"
              << "Global:\n"
              << "  -type {mnist|sift}  -d <input>  -q <query>  -N <int>  -R <float>  -o <output>\n"
              << "  -range {true|false} -seed <int>\n\n"
              << "Choose one:\n"
              << "  -lsh       -k <int> -L <int> -w <float>\n"
              << "  -hypercube -kproj <int> -w <float> -M <int> -probes <int>\n"
              << "  -ivfflat   -kclusters <int> -nprobe <int>\n"
              << "  -ivfpq     -kclusters <int> -nprobe <int> -M <int> -nbits <int>\n";
}

bool CLI::parse(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](int n){ if (i+n >= argc) { print_help(); return false; } return true; };

        if (a == "-h" || a == "--help") { print_help(); return false; }
        else if (a == "-type" && need(1)) {
            std::string t = argv[++i];
            if (t == "mnist") {
                type = MNIST;
                kclusters = kclusters_mnist;
            }
            else if (t == "sift") {
                type = SIFT;
                kclusters = kclusters_sift;
            }
            else { std::cerr << "type must be mnist or sift.\n"; return false; }
        }
        else if (a == "-d" && need(1)) { dfile = argv[++i]; }
        else if (a == "-N" && need(1)) { N = static_cast<uint32_t>(std::stoul(argv[++i])); }
        else if (a == "-seed" && need(1)) { seed = static_cast<uint32_t>(std::stoul(argv[++i])); }

        else if (a == "-lsh") { method = LSH; }
        else if (a == "-hypercube") { method = HYPERCUBE; }
        else if (a == "-ivfflat") { method = IVFFLAT; }
        else if (a == "-ivfpq") { method = IVFPQ; }

        else if (a == "-k" && need(1)) { k = std::stoi(argv[++i]); }
        else if (a == "-L" && need(1)) { L = std::stoi(argv[++i]); }
        else if (a == "-w" && need(1)) { w = std::stof(argv[++i]); }

        else if (a == "-kproj" && need(1)) { kproj = std::stoi(argv[++i]); }
        else if (a == "-M" && need(1)) { M = std::stoi(argv[++i]); }
        else if (a == "-probes" && need(1)) { probes = std::stoi(argv[++i]); }

        else if (a == "-kclusters" && need(1)) { kclusters = std::stoi(argv[++i]); }
        else if (a == "-nprobe" && need(1)) { nprobe = std::stoi(argv[++i]); }

        else if (a == "-nbits" && need(1)) { nbits = std::stoi(argv[++i]); }
        else { std::cerr << "Unknown/Bad arg: " << a << "\n"; return false; }
    }

    if (method == NONE) { std::cerr << "Select exactly one method flag.\n"; return false; }

    if (dfile.empty()) { std::cerr << "Provide -d and -q files.\n"; return false; }

    return true;
}
