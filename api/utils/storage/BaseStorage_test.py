import unittest
from . import Storage, S3Storage, HTTPStorage


class BaseStorageTest(unittest.TestCase):
    def test_get_filename(self):
        storage = Storage("http://host.com/dir/file.tar.zst")
        self.assertEqual(storage.get_filename(), "file.tar.zst")

    class Download_and_extract(unittest.TestCase):
        def test_file_only(self):
            storage = Storage("http://host.com/dir/file.bin")
            result = storage.download_and_extract(dry_run=True)
            self.assertEqual(result, "file.bin")

        def test_file_archive(self):
            storage = Storage("http://host.com/dir/file.tar.zst")
            result, base, ext, subext = storage.download_and_extract(dry_run=True)
            self.assertEqual(result, "file")
            self.assertEqual(base, "file")
            self.assertEqual(ext, "tar")
            self.assertEqual(subext, "zst")
