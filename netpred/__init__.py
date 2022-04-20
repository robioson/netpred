import logging
import os
from pathlib import Path

# set global logging configuration
logging.basicConfig(
    format='%(asctime)s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# paths will be relative to the `data` directory alongside the package
data_files_directory = Path(__file__).parent.parent / 'data'
os.makedirs(data_files_directory, exist_ok=True)
os.chdir(data_files_directory)
