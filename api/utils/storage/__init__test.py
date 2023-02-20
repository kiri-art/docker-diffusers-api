import unittest
from . import Storage, S3Storage, HTTPStorage


class StorageTest(unittest.TestCase):
    def test_url_s3(self):
        storage = Storage("s3://hostname:9000")
        self.assertTrue(isinstance(storage, S3Storage))

    def test_url_http(self):
        storage = Storage("http://hostname:9000")
        self.assertTrue(isinstance(storage, HTTPStorage))

    def test_no_match_raise(self):
        with self.assertRaises(RuntimeError):
            storage = Storage("not_a_url")

    def test_no_match_no_raise(self):
        storage = Storage("not_a_url", no_raise=True)
        self.assertIsNone(storage)
