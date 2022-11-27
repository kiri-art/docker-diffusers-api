import unittest
import os
from S3Storage import S3Storage, AWS_S3_ENDPOINT_URL


class S3StorageTest(unittest.TestCase):
    def test_endpoint_only_s3(self):
        storage = S3Storage("s3://hostname:9000")
        self.assertEqual(storage.endpoint_url, "https://hostname:9000")
        self.assertEqual(storage.bucket_name, None)
        self.assertEqual(storage.path, "")

    def test_endpoint_only_http_s3(self):
        storage = S3Storage("http+s3://hostname:9000")
        self.assertEqual(storage.endpoint_url, "http://hostname:9000")
        self.assertEqual(storage.bucket_name, None)
        self.assertEqual(storage.path, "")

    def test_endpoint_only_https_s3(self):
        storage = S3Storage("https+s3://hostname:9000")
        self.assertEqual(storage.endpoint_url, "https://hostname:9000")
        self.assertEqual(storage.bucket_name, None)
        self.assertEqual(storage.path, "")

    def test_bucket_only(self):
        storage = S3Storage("s3:///bucket")
        self.assertEqual(storage.endpoint_url, AWS_S3_ENDPOINT_URL)
        self.assertEqual(storage.bucket_name, "bucket")
        self.assertEqual(storage.path, "")

    def test_url_with_bucket_and_file_only(self):
        storage = S3Storage("s3:///bucket/file")
        self.assertEqual(storage.endpoint_url, AWS_S3_ENDPOINT_URL)
        self.assertEqual(storage.bucket_name, "bucket")
        self.assertEqual(storage.path, "file")

    def test_full_url_with_subdirectory(self):
        storage = S3Storage("s3://host/bucket/path/file")
        self.assertEqual(storage.endpoint_url, "https://host")
        self.assertEqual(storage.bucket_name, "bucket")
        self.assertEqual(storage.path, "path/file")


if __name__ == "__main__":
    unittest.main()
