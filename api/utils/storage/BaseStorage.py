import os
import re
import subprocess
from abc import ABC, abstractmethod


class BaseArchive(ABC):
    def __init__(self, path):
        self.path = path

    def extract(self):
        print("TODO")

    def splitext(self):
        base, ext = os.path.splitext(self.path)
        base, subext = os.path.splitext(base)
        return base, ext, subext


class TarZstdArchive(BaseArchive):
    @staticmethod
    def test(path):
        return re.search(r"\.tar\.zstd?$", path)

    def extract(self, dir, dry_run=False):
        if not dir:
            dir = os.path.dirname(self.path)
        base, ext, subext = self.splitext()
        dir = os.path.join(dir, base)

        if not dry_run:
            os.mkdir(dir)
            subprocess.run(
                [
                    "tar",
                    "--use-compress-program=unzstd",
                    "-C",
                    dir,
                    "-xvf",
                    self.path,
                ],
                check=True,
            )
            os.remove(self.path)

        return dir  # , base, ext, subext


archiveClasses = [TarZstdArchive]


def Archive(path, **kwargs):
    for ArchiveClass in archiveClasses:
        if ArchiveClass.test(path):
            return ArchiveClass(path, **kwargs)


class BaseStorage(ABC):
    @staticmethod
    @abstractmethod
    def test(url):
        return re.search(r"^https?://", url)

    def __init__(self, url, **kwargs):
        self.url = url

    def splitext(self):
        base, ext = os.path.splitext(self.url)
        base, subext = os.path.splitext(base)
        return base, ext, subext

    def get_filename(self):
        return self.url.split("/").pop()

    @abstractmethod
    def download_file(self, dest):
        """Download the file to `dest`"""
        pass

    def download_and_extract(self, fname, dry_run=False):
        """
        Downloads the file, and if it's an archive, extract it too.  Returns
        the filename if not, or directory name (fname without extension) if
        it was.
        """
        if not fname:
            fname = self.get_filename()

        dir = None
        archive = Archive(fname, dry_run=dry_run)
        if archive:
            # TODO, streaming pipeline
            self.download_file(fname)
            return archive.extract()
        else:
            self.download_file(fname)
            return fname
