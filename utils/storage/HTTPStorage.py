import re
import os
import time
import requests
from tqdm import tqdm


def get_now():
    return round(time.time() * 1000)


class HTTPStorage:
    def __init__(self, url, path=""):
        self.url = url

        if url.startswith("s3://"):
            url = "https://" + url[5:]
        elif url.startswith("http+s3://"):
            url = "http" + url[7:]
        elif url.startswith("https+s3://"):
            url = "https" + url[8:]

        s3_dest = re.match(
            "^(?P<endpoint>https?://[^/]*)(/(?P<bucket>[^/]+))?(/(?P<path>.*))?$",
            url,
        ).groupdict()

        if not s3_dest["endpoint"] or s3_dest["endpoint"].endswith("//"):
            s3_dest["endpoint"] = AWS_S3_ENDPOINT_URL
        if not s3_dest["bucket"]:
            s3_dest["bucket"] = AWS_S3_DEFAULT_BUCKET
        if not s3_dest["path"] or s3_dest["path"] == "":
            s3_dest["path"] = path

        self.endpoint_url = s3_dest["endpoint"]
        self.bucket_name = s3_dest["bucket"]
        self.path = s3_dest["path"]

        self._s3 = None
        self._bucket = None
        print("self.endpoint_url", self.endpoint_url)

    def upload_file(self, source, dest):
        raise RuntimeError("HTTP PUT not implemented yet")

    def download_file(self, fname):
        print(f"Downloading {self.url} to {fname}...")
        resp = requests.get(self.url, stream=True)
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
