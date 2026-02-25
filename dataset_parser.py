import struct
from typing import Tuple, Optional

import numpy as np


R_MNIST_DEFAULT = 2000.0
R_SIFT_DEFAULT = 2800.0


def load_mnist_images(path: str) -> np.ndarray:
    """
    Read idx3-ubyte MNIST images into numpy array [n, d] (float32).
    """
    with open(path, "rb") as f:
        header = f.read(16)
        if len(header) < 16:
            raise ValueError(f"Bad MNIST file: {path}")

        magic, nimg, nrows, ncols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(
                f"Bad MNIST magic number (expected 2051, got {magic}) in {path}"
            )

        d = nrows * ncols
        data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        if arr.size % d != 0:
            raise ValueError(f"MNIST data size not divisible by {d} in {path}")
        arr = arr.reshape(-1, d).astype(np.float32)
    return arr


def load_sift_fvecs(path: str) -> np.ndarray:
    """
    Read SIFT .fvecs file into numpy array [n, d] (float32).
    Format: int32 d, followed by d float32 (little endian), repeated.
    """
    vectors = []
    with open(path, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # EOF
            if len(dim_bytes) < 4:
                raise ValueError(f"Truncated fvecs file (dim) in {path}")
            (d,) = struct.unpack("<i", dim_bytes)
            vec_bytes = f.read(4 * d)
            if len(vec_bytes) < 4 * d:
                raise ValueError(f"Truncated fvecs file (vector) in {path}")
            v = np.frombuffer(vec_bytes, dtype="<f4")  # little-endian float32
            vectors.append(v.copy())
    if not vectors:
        raise ValueError(f"No vectors read from {path}")
    return np.stack(vectors, axis=0)


def load_dataset(
    data_path: str,
    query_path: Optional[str],
    dtype: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Return (data, query) as numpy arrays [n, d] (float32).
    dtype: 'mnist' ή 'sift'.
    """
    dtype = dtype.lower()
    if dtype == "mnist":
        data = load_mnist_images(data_path)
        queries = load_mnist_images(query_path) if query_path is not None else None
    elif dtype == "sift":
        data = load_sift_fvecs(data_path)
        queries = load_sift_fvecs(query_path) if query_path is not None else None
    else:
        raise ValueError("dtype must be 'mnist' or 'sift'")
    return data, queries
