# NETPRED

The source code for NETPRED (an interpretable model for secondary structure prediction) can be
found in the [`netpred`](netpred) subdirectory. Notebooks to analyse the model's behaviour can
be found in the [`notebooks`](notebooks) directory.

### Requirements

* Python 3.9+
* [Poetry](https://python-poetry.org/docs)
* An internet connection to download dependencies and data files
* At least 16GB of RAM for training, and around 15GB of disk space for files and dependencies
* Tested on GNU/Linux; other platforms will probably work, but they may not.

### Installing dependencies

* Navigate to this directory
* Run `poetry install`.

### Using the software

* [`config.py`](netpred/config.py) can be edited to configure embedding, data sources, and so on
* Once you have created the Poetry environment, run `make train` to begin training. All data
files will be downloaded and extracted automatically
* To browse the notebooks, run `make analyse`.
