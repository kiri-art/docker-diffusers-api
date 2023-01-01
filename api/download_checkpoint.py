import os
from utils import Storage

CHECKPOINT_URL = os.environ.get("CHECKPOINT_URL", None)
CHECKPOINT_DIR = "/root/.cache/checkpoints"


def main(checkpoint_url: str):
    if not os.path.isdir(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    fname = CHECKPOINT_DIR + "/" + checkpoint_url.split("/").pop()
    if not os.path.isfile(fname):
        storage = Storage(checkpoint_url)
        storage.download_file(fname)


if __name__ == "__main__":
    if CHECKPOINT_URL:
        main(CHECKPOINT_URL)
