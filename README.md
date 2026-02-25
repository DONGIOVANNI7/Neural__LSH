# Neural LSH Vector Search in Python

**Authors:** Ιωάννης Νικολόπουλος (sdi2100292), Ιωάννης Πετράκης (sdi1900155)

Neural LSH implementation for high-dimensional Approximate Nearest Neighbor (ANN) search using PyTorch, NumPy, and KaHIP. Builds on Assignment 1's C++ ANN methods to construct k-NN graphs for efficient space partitioning and MLP-based query routing.

For a detailed analysis visit the pdf reports!

## Quick Start

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt  # numpy, torch, kahip

# Ensure C++ backend from ANN_Vector_Search is built: knn/bin/approx_knn

# Build index (MNIST example)
python nlsh_build.py \
  -d data/mnist-data.dat \
  -i indexes/mnist_nlsh.pt \
  -type mnist \
  --knn 10 --knn_method ivfflat \
  -m 100 --layers 3 --nodes 256 --epochs 10

# Search
python nlsh_search.py \
  -d data/mnist-data.dat \
  -q data/mnist-query.dat \
  -i indexes/mnist_nlsh.pt \
  -o out/mnist_results.txt \
  -type mnist -N 10 -T 5 -R 2000 --range true

#How It Works
1)k-NN Graph Construction: Uses Assignment 1 methods (LSH/Hypercube/IVFFlat/IVFPQ) via approx_knn.
2)Graph Partitioning: KaHIP partitions the weighted undirected graph into m balanced bins.
3)MLP Training: Neural classifier learns to predict partition labels from raw vectors.
4)Multi-Probe Search: Query routed to top-T bins by softmax probability; exact search on candidates.

#Key Parameters

| Phase      | Parameter              | Description             | Default             |
| ---------- | ---------------------- | ----------------------- | ------------------- |
| **Build**  | `--knn`                | Graph neighbors         | 10                  |
|            | `-m`                   | Number of partitions    | 100                 |
|            | `--kahip_mode`         | 0=FAST, 1=ECO, 2=STRONG | 2                   |
|            | `--layers` / `--nodes` | MLP architecture        | 3 layers / 64 nodes |
| **Search** | `-N`                   | Neighbors returned      | 10                  |
|            | `-T`                   | Top-T bins probed       | 5                   |
|            | `--batch_size`         | Inference batch         | 256                 |
