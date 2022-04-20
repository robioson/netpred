import json
import os
from logging import info, debug
from random import shuffle

from torch import tensor
from torch.utils import data

from .config import (
    DATASET,
    PROTEINNET_URL,
    ASTRAL_SECONDARY_STRUCTURES_URL,
    PDB_SECONDARY_STRUCTURES_URL,
    BATCH_SIZE
)
from .download import download_and_unpack, download_and_extract
from .sequences import dssp_to_3_state, create_windows


def load_secondary_structures() -> dict[str, tensor]:
    """Download secondary structure data (not part of main ProteinNet download), and convert
    DSSP sequences to 3-state encodes."""
    pdb = json.loads(download_and_extract(PDB_SECONDARY_STRUCTURES_URL))
    astral = json.loads(download_and_extract(ASTRAL_SECONDARY_STRUCTURES_URL))
    structures = pdb | astral

    return {id_: dssp_to_3_state(sequences['DSSP'])
            for id_, sequences in structures.items()}


def read_protein_data(path: str) -> dict[str, tuple[str, tensor, tensor]]:
    """Download ProteinNet (if necessary), then parse a ProteinNet "human readable" record at `path`.
    The output is a dict mapping protein ID to a tuple of (primary, pssm, secondary).
    Where:
        primary is the primary sequence as a string.
        pssm is the scoring matrix as a 21xlen(primary) tensor. The extra (last) dimension
            is all zeros. This is used later in the windowing step.
        secondary is the 3-state secondary sequence as a 1xlen(primary) tensor. This is
            mapped from the DSSP that ProteinNet provides natively.
    The format is described here:
        <https://github.com/aqlaboratory/proteinnet/blob/master/docs/proteinnet_records.md>."""
    download_and_unpack(PROTEINNET_URL)

    info(f'Parsing {path}...')

    with open(path) as f:
        file = f.read()

    result = {}

    # proteins are separated by blank line
    proteins = file.split('\n\n')
    for p in proteins:
        # empty section marks end of file
        if not p:
            break

        lines = p.split('\n')

        id_ = lines[1].split('#', maxsplit=1)[-1]

        primary = lines[3]

        try:
            secondary = secondary_structures[id_]
        except KeyError:
            # some sequences in ProteinNet are not in the DSSP dataset so ignore these
            debug(f'{id_!r} not in DSSP data')
            continue

        # the "evolutionary" section consists of 21 rows of tab-separated floats.
        # the first 20 dimensions are the PSSM, followed by information content.
        evolutionary = lines[5:26]

        pssm = [[float(f) for f in v.split('\t')] for v in evolutionary]

        result[id_] = (primary, tensor(pssm), secondary)

    info(f'Loaded {len(result)} sequences from {path}.')
    return result


secondary_structures = load_secondary_structures()


class ProteinDataset(data.Dataset):
    def __init__(self, file: str):
        path = os.path.join(DATASET, file)
        protein_data = read_protein_data(path)

        info(f'Creating windows for {path} ...')
        samples = []
        for (primary, pssm, secondary) in protein_data.values():
            samples.extend(create_windows(pssm, secondary))
        info(f'Windowing complete for {path} (created {len(samples)} windows).')

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def shuffle(self):
        """Shuffle the dataset. Useful as an option in analysis."""
        shuffle(self.samples)


class ProteinDataLoader(data.DataLoader):
    def __init__(self, file: str, batch_size: int = BATCH_SIZE):
        super().__init__(ProteinDataset(file), batch_size=batch_size)
