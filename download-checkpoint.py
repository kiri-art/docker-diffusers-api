import os
from utils import Storage

CHECKPOINT_URL = os.environ.get("CHECKPOINT_URL", None)

if __name__ == "__main__":
    if CHECKPOINT_URL:
        CHECKPOINT_DIR = "/root/.cache/checkpoints"
        if not os.path.isdir(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        fname = CHECKPOINT_DIR + "/" + CHECKPOINT_URL.split("/").pop()
        if not os.path.isfile(fname):
            storage = Storage(CHECKPOINT_URL)
            storage.download_file(fname)
