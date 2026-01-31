import logging
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = PROJECT_ROOT / "data"
ZIP_PATH = DATA_DIR / "ml-1m.zip"


def download_movielens() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    if (DATA_DIR / "ml-1m" / "ratings.dat").exists():
        logger.info("MovieLens 1M already downloaded")
        return

    try:
        logger.info(f"Downloading MovieLens 1M from {MOVIELENS_URL}")
        urllib.request.urlretrieve(MOVIELENS_URL, ZIP_PATH)
        logger.info(f"Downloaded to {ZIP_PATH}")

        logger.info("Extracting...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:
            zip_file.extractall(DATA_DIR)

        logger.info(f"Extracted to {DATA_DIR / 'ml-1m'}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to download MovieLens dataset: {e}") from e
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Downloaded file is corrupted: {e}") from e
    finally:
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_movielens()
