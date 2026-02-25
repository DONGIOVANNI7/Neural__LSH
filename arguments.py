import argparse

from graph_utils import KnnMethods

def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural LSH index builder")

    parser.add_argument(
        "-d",
        type=str,
        required=True,
        help="Input dataset file (base)"
    )
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        help="Index path (e.g. nlsh_index.pt)"
    )
    parser.add_argument(
        "-type",
        type=str,
        required=True,
        choices=["mnist", "sift"],
        help="Dataset type",
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=10,
        help="k for k-NN graph (default: 10)"
    )

    parser.add_argument(
        "--knn_method",
        type=str,
        default="ivfflat",
        help="knn method (lsh, hypercube, ivfflat, ivfpq)"
    )

    parser.add_argument(
        "-m",
        type=int,
        default=100,
        help="Number of partitions m (default: 100)"
    )
    parser.add_argument(
        "--imbalance",
        type=float,
        default=0.03,
        help="KaHIP imbalance ε (default: 0.03)",
    )
    parser.add_argument(
        "--kahip_mode",
        type=int,
        default=2,
        help="KaHIP mode: 0=FAST,1=ECO,2=STRONG (default: 2)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=3,
        help="Number of MLP layers (default: 3)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=64,
        help="Hidden units per layer (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="MLP epochs (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="MLP batch size (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="MLP learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1)",
    )
    return parser.parse_args()

def search_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neural LSH search")

    parser.add_argument(
        "-d",
        type=str,
        required=True,
        help="Input dataset file (base)"
    )
    parser.add_argument(
        "-q",
        type=str,
        required=True,
        help="Query dataset file"
    )
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        help="Index path (from nlsh_build)"
    )
    parser.add_argument(
        "-o",
        type=str,
        required=True,
        help="Output file"
    )
    parser.add_argument(
        "-type",
        type=str,
        required=True,
        choices=["mnist", "sift"],
        help="Dataset type",
    )
    parser.add_argument(
        "-N",
        type=int,
        default=1,
        help="Number of nearest neighbors (default: 1)"
    )
    parser.add_argument(
        "-R",
        type=float,
        default=None,
        help="Range radius R (default: 2000 for MNIST, 2800 for SIFT)",
    )
    parser.add_argument(
        "-T",
        type=int,
        default=5,
        help="Number of bins to probe (multi-probe) (default: 5)",
    )
    parser.add_argument(
        "-range",
        dest="do_range",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Whether to perform range search (default: true)",
    )
    return parser.parse_args()
