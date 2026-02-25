import argparse
import numpy as np

from dataset_parser import load_dataset
from graph_utils import brute_force_knn

def store_knn(knn: np.ndarray, path: str) -> None:
    with open(path, "wb") as f:
        np.save(f, knn)


def load_knn(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        knn = np.load(f)

    return knn

def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute knn for MNIST or SIFT dataset")

    parser.add_argument(
        "-d",
        type=str,
        required=True,
        help="Dataset path"
    )
    parser.add_argument(
        "-o",
        type=str,
        required=True,
        help="Output path"
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["mnist", "sift"],
        help="Dataset type"
    )
    parser.add_argument(
        "-k",
        type=int,
        required=False,
        default=100,
        help="Number of neighbors"
    )
    args = parser.parse_args()

    dataset, _ = load_dataset(args.d, None, args.type)
    knn = brute_force_knn(dataset, args.k)
    store_knn(knn, args.o)

if __name__ == "__main__":
    main()
