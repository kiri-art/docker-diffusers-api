import os
import re
from .S3Storage import S3Storage


def Storage(url):
    if re.search("^(https?\+)?s3://", url):
        return S3Storage(url)
    raise RuntimeError("No storage handler for: " + url)
