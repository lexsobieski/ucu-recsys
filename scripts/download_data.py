"""Download MovieLens 1M dataset."""

import urllib.request
import zipfile
from pathlib import Path

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = Path("data/raw")


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "ml-1m.zip"

    if (DATA_DIR / "ratings.dat").exists():
        print("Data already exists, skipping download.")
        return

    print(f"Downloading {DATA_URL}...")
    urllib.request.urlretrieve(DATA_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith(".dat"):
                # Extract flat (no ml-1m/ subfolder)
                filename = Path(member).name
                with zf.open(member) as src, open(DATA_DIR / filename, "wb") as dst:
                    dst.write(src.read())

    zip_path.unlink()
    print("Done.")


if __name__ == "__main__":
    download()
