import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "ml-1m"
ZIP_PATH = DATA_DIR / "ml-1m.zip"
RATINGS_FILENAME = "ratings.dat"


def download_movielens() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    if (DATASET_DIR / RATINGS_FILENAME).exists():
        print("MovieLens 1M already downloaded")
        return

    try:
        print(f"Downloading MovieLens 1M from {MOVIELENS_URL}")
        urllib.request.urlretrieve(
            url=MOVIELENS_URL,
            filename=ZIP_PATH
        )
        print(f"Downloaded to {ZIP_PATH}")

        print("Extracting...")
        with zipfile.ZipFile(file=ZIP_PATH, mode="r") as zip_file:
            zip_file.extractall(path=DATA_DIR)
        print(f"Extracted to {DATASET_DIR}")

    except urllib.error.URLError as error:
        raise RuntimeError(f"Failed to download MovieLens dataset: {error}") from error

    except zipfile.BadZipFile as error:
        raise RuntimeError(f"Downloaded file is corrupted: {error}") from error

    finally:
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()


if __name__ == "__main__":
    download_movielens()
