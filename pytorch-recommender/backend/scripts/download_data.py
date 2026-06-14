"""Standalone helper to pre-download the MovieLens dataset.

Usage:  python -m scripts.download_data            # ml-100k (default)
        python -m scripts.download_data ml-1m

Training also downloads automatically on first run; this is just for warming the
data cache (e.g. in a Docker build) without training.
"""

import sys

from app.config import settings
from app.data import download_movielens


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else settings.dataset
    path = download_movielens(settings.data_dir, dataset)
    print(f"Downloaded {dataset} -> {path}")


if __name__ == "__main__":
    main()
