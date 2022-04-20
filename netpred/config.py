from .embedding import selected_encoding

########################################################################################################################
# MODEL CONFIGURATION

# if True, the network will be trained on VHSE-embedded PSSMs.
# if False, PSSMs will be used directly.
EMBED = False

PSSM_ROWS = 2 + (selected_encoding.size(1) if EMBED else 20)

# number of residues in each window. must be odd.
WINDOW_SIZE = 27
assert WINDOW_SIZE % 2

# index of central residue in window.
WINDOW_CENTRE = WINDOW_SIZE // 2

########################################################################################################################
# DATA CONFIGURATION

# ProteinNet component that will be downloaded and used for training/validation.
DATASET = 'casp11'

# Training set file within DATASET to use.
TRAINING_SET = 'training_70'

# Validation set file within DATASET to use.
VALIDATION_SET = 'validation'

# locations of ProteinNet.
PROTEINNET_URL = f'https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/{DATASET}.tar.gz'
PDB_SECONDARY_STRUCTURES_URL = 'https://www.dropbox.com/s/raw/sne2ak1woy1lrqr/full_protein_dssp_annotations.json.gz'
ASTRAL_SECONDARY_STRUCTURES_URL = 'https://www.dropbox.com/s/raw/59y3nud4rixombf/single_domain_dssp_annotations.json.gz'

########################################################################################################################
# TRAINING CONFIGURATION

# Print loss after this many mini-batches.
PRINT_EVERY = 2000

# Default mini-batch size for data loaders.
BATCH_SIZE = 512
