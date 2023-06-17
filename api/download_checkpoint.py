import os
from utils import Storage

CHECKPOINT_URL = os.environ.get("CHECKPOINT_URL", None)
CHECKPOINT_DIR = "/root/.cache/checkpoints"


def main(checkpoint_url: str):
    if not os.path.isdir(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    storage = Storage(checkpoint_url)
    storage_query_fname = storage.query.get("fname")
    if storage_query_fname:
        fname = storage_query_fname[0]
    else:
        fname = checkpoint_url.split("/").pop()
    path = os.path.join(CHECKPOINT_DIR, fname)

    if not os.path.isfile(path):
        storage.download_file(path)

    return path


if __name__ == "__main__":
    if CHECKPOINT_URL:
        main(CHECKPOINT_URL)
