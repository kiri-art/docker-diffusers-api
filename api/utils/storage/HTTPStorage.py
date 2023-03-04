import re
import os
import time
import requests
from tqdm import tqdm
from .BaseStorage import BaseStorage


def get_now():
    return round(time.time() * 1000)


class HTTPStorage(BaseStorage):
    @staticmethod
    def test(url):
        return re.search(r"^https?://", url)

    def __init__(self, url, **kwargs):
        self.url = url

    def upload_file(self, source, dest):
        raise RuntimeError("HTTP PUT not implemented yet")

    def download_file(self, fname):
        print(f"Downloading {self.url} to {fname}...")
        resp = requests.get(self.url, stream=True)
        total = int(resp.headers.get("content-length", 0))
        content_disposition = resp.headers["content-disposition"]
        filename_search = re.search('filename="(.+)"', content_disposition)
        if filename_search:
            self.filename = filename_search.group(1)
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
