import os
import requests
from tqdm import tqdm

CHECKPOINT_URL = os.environ.get("CHECKPOINT_URL", None)


def download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    # Can also replace 'file' with a io.BytesIO object
    with open(fname, "wb") as file, tqdm(
        desc="Downloading",
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


if __name__ == "__main__":
    if CHECKPOINT_URL:
        CHECKPOINT_DIR = "/root/.cache/checkpoints"
        if not os.path.isdir(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        fname = CHECKPOINT_DIR + "/" + CHECKPOINT_URL.split("/").pop()
        if not os.path.isfile(fname):
            print("Downloading " + CHECKPOINT_URL)
            download(url=CHECKPOINT_URL, fname=fname)
