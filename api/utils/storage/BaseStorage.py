import os
import re
import subprocess
from abc import ABC, abstractmethod
import xtarfile as tarfile


class BaseArchive(ABC):
    def __init__(self, path, status=None):
        self.path = path
        self.status = status

    def updateStatus(self, type, progress):
        if hasattr(self, "status"):
            self.status.update(type, progress)

    def extract(self):
        print("TODO")

    def splitext(self):
        base, ext = os.path.splitext(self.path)
        base, subext = os.path.splitext(base)
        return base, ext, subext


class TarArchive(BaseArchive):
    @staticmethod
    def test(path):
        return re.search(r"\.tar", path)

    def extract(self, dir, dry_run=False):
        self.updateStatus("extract", 0)
        if not dir:
            dir = os.path.dirname(self.path)
        base, ext, subext = self.splitext()
        dir = os.path.join(dir, base)

        if not dry_run:
            os.mkdir(dir)

            def track_progress(tar):
                i = 0
                members = tar.getmembers()
                for member in members:
                    i += 1
                    self.updateStatus("extract", i / len(members))
                    yield member

            with tarfile.open(self.path, "r") as tar:
                tar.extractall(path=dir, members=track_progress(tar))
                tar.close()
            subprocess.run(["ls", "-l"])
            os.remove(self.path)

        self.updateStatus("extract", 1)
        return dir  # , base, ext, subext


archiveClasses = [TarArchive]


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
        self.status = kwargs.get("status", None)

    def updateStatus(self, type, progress):
        if hasattr(self, "status"):
            self.status.update(type, progress)

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

    def download_and_extract(self, fname, dir=None, dry_run=False):
        """
        Downloads the file, and if it's an archive, extract it too.  Returns
        the filename if not, or directory name (fname without extension) if
        it was.
        """
        if not fname:
            fname = self.get_filename()

        archive = Archive(fname, status=self.status)
        if archive:
            # TODO, streaming pipeline
            self.download_file(fname)
            return archive.extract(dir)
        else:
            self.download_file(fname)
            return fname
