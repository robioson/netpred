"""Utility functions to download and extract/unpack files."""
import gzip
from logging import info
from os.path import basename, exists
from shutil import unpack_archive
from urllib.request import urlretrieve


def download_file(url: str) -> str:
    """Download a file if it does not exist on the filesystem, otherwise do nothing."""
    filename = basename(url)
    if not exists(filename):
        info(f'Downloading <{url}>')
        urlretrieve(url, filename)
    return filename


def download_and_unpack(url: str):
    filename = download_file(url)
    if not exists(filename[:filename.index('.')]):
        info(f'Unpacking {filename}')
        unpack_archive(filename, '.')


def download_and_extract(url: str) -> str:
    filename = download_file(url)
    info(f'Decompressing {filename}')
    with gzip.open(filename, 'rb') as f:
        return f.read().decode('ascii')
