import os
import re
from .S3Storage import S3Storage
from .HTTPStorage import HTTPStorage


def Storage(url, **kwargs):
    if re.search(r"^(https?\+)?s3://", url):
        return S3Storage(url, **kwargs)

    if re.search(r"^https?://", url):
        return HTTPStorage(url, **kwargs)

    raise RuntimeError("No storage handler for: " + url)
