# Common setup code for analysis notebooks.

import torch
import numpy as np
import matplotlib.pyplot as plt

from Bio.Alphabet.IUPAC import IUPACProtein
from Bio.SeqUtils import seq3

from .data import ProteinDataset
from .networks import net
from .sequences import DECODE_3_STATE, ENCODE_3_STATE
from .config import *

# load trained weights and globally disable gradient computation
net.load_state_dict(torch.load(net.pth_name))
torch.autograd.set_grad_enabled(False)

# three-letter amino acid codes, in order of their one-letter code
# this is the order that ProteinNet has its PSSMs in (I think...)
amino_names = tuple(map(seq3, IUPACProtein.letters))

# add another row name for the terminus indicator
pssm_rows = (*amino_names, 'TER')

# class labels
tick_labels = ['α', 'β', 'L']

test_set = ProteinDataset('validation')
