import os
import re
from .S3Storage import S3Storage
from .HTTPStorage import HTTPStorage

classes = [S3Storage, HTTPStorage]


def Storage(url, no_raise=False, **kwargs):
    for StorageClass in classes:
        if StorageClass.test(url):
            return StorageClass(url, **kwargs)

    if no_raise:
        return None
    else:
        raise RuntimeError("No storage handler for: " + url)
