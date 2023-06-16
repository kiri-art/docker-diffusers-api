import re
import os
import time
import requests
from tqdm import tqdm
from .BaseStorage import BaseStorage
import urllib.parse


def get_now():
    return round(time.time() * 1000)


class HTTPStorage(BaseStorage):
    @staticmethod
    def test(url):
        return re.search(r"^https?://", url)

    def __init__(self, url, **kwargs):
        super().__init__(url, **kwargs)
        parts = self.url.split("#", 1)
        self.url = parts[0]
        if len(parts) > 1:
            self.query = urllib.parse.parse_qs(parts[1])

    def upload_file(self, source, dest):
        raise RuntimeError("HTTP PUT not implemented yet")

    def download_file(self, fname):
        print(f"Downloading {self.url} to {fname}...")
        resp = requests.get(self.url, stream=True)
        total = int(resp.headers.get("content-length", 0))
        content_disposition = resp.headers.get("content-disposition")
        if content_disposition:
            filename_search = re.search('filename="(.+)"', content_disposition)
            if filename_search:
                self.filename = filename_search.group(1)
        else:
            print("Warning: content-disposition header is not found in the response.")
        # Can also replace 'file' with a io.BytesIO object
        with open(fname, "wb") as file, tqdm(
            desc="Downloading",
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            total_written = 0
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
                total_written += size
                self.updateStatus("download", total_written / total)
