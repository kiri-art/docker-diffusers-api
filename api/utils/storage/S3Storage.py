import boto3
import botocore
import re
import os
import time
from tqdm import tqdm
from botocore.client import Config


AWS_S3_ENDPOINT_URL = os.environ.get("AWS_S3_ENDPOINT_URL", None)
AWS_S3_DEFAULT_BUCKET = os.environ.get("AWS_S3_DEFAULT_BUCKET", None)
if AWS_S3_ENDPOINT_URL == "":
    AWS_S3_ENDPOINT_URL = None
if AWS_S3_DEFAULT_BUCKET == "":
    AWS_S3_DEFAULT_BUCKET = None


def get_now():
    return round(time.time() * 1000)


class S3Storage:
    def __init__(self, url, **kwargs):
        self.url = url

        if url.startswith("s3://"):
            url = "https://" + url[5:]
        elif url.startswith("http+s3://"):
            url = "http" + url[7:]
        elif url.startswith("https+s3://"):
            url = "https" + url[8:]

        s3_dest = re.match(
            r"^(?P<endpoint>https?://[^/]*)(/(?P<bucket>[^/]+))?(/(?P<path>.*))?$",
            url,
        ).groupdict()

        if not s3_dest["endpoint"] or s3_dest["endpoint"].endswith("//"):
            s3_dest["endpoint"] = AWS_S3_ENDPOINT_URL
        if not s3_dest["bucket"]:
            s3_dest["bucket"] = AWS_S3_DEFAULT_BUCKET
        if not s3_dest["path"] or s3_dest["path"] == "":
            s3_dest["path"] = kwargs.get("default_path", "")

        self.endpoint_url = s3_dest["endpoint"]
        self.bucket_name = s3_dest["bucket"]
        self.path = s3_dest["path"]

        self._s3resource = None
        self._s3client = None
        self._bucket = None
        print("self.endpoint_url", self.endpoint_url)

    def s3resource(self):
        if self._s3resource:
            return self._s3resource

        self._s3 = boto3.resource(
            "s3",
            endpoint_url=self.endpoint_url,
            config=Config(signature_version="s3v4"),
        )
        return self._s3

    def s3client(self):
        if self._s3client:
            return self._s3client

        self._s3client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            config=Config(signature_version="s3v4"),
        )
        return self._s3client

    def bucket(self):
        if self._bucket:
            return self._bucket

        self._bucket = self.s3resource().Bucket(self.bucket_name)
        return self._bucket

    def upload_file(self, source, dest):
        if not dest:
            dest = self.path

        upload_start = get_now()
        file_size = os.stat(source).st_size
        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Uploading") as bar:
            result = self.bucket().upload_file(
                Filename=source,
                Key=dest,
                Callback=lambda bytes_transferred: bar.update(bytes_transferred),
            )
        print(result)
        upload_total = get_now() - upload_start

        return {"$time": upload_total}

    def download_file(self, dest):
        if not dest:
            dest = self.path.split("/").pop()
        print(f"Downloading {self.url} to {dest}...")
        object = self.s3resource().Object(self.bucket_name, self.path)
        object.load()

        with tqdm(
            total=object.content_length, unit="B", unit_scale=True, desc="Downloading"
        ) as bar:
            object.download_file(
                Filename=dest,
                Callback=lambda bytes_transffered: bar.update(bytes_transffered),
            )

    def file_exists(self):
        # res = self.s3client().list_objects_v2(
        #    Bucket=self.bucket_name, Prefix=self.path, MaxKeys=1
        # )
        # return "Contents" in res
        object = self.s3resource().Object(self.bucket_name, self.path)
        try:
            object.load()
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] == "404":
                return False
            else:
                raise
        return True
