"""Utilities for working on sequences."""
import torch
from torch import tensor
from torch.nn.functional import pad

from .config import EMBED, WINDOW_SIZE, WINDOW_CENTRE
from .embedding import embed_pssm

embed_pssm = embed_pssm if EMBED else lambda x: x

DECODE_3_STATE: dict[int, str] = dict(enumerate('HEC'))
# noinspection PyTypeChecker
ENCODE_3_STATE: dict[str, int] = dict(map(reversed, DECODE_3_STATE.items()))

DSSP_TO_3_STATE = {
    'E': ENCODE_3_STATE['E'],
    'B': ENCODE_3_STATE['E'],
    'H': ENCODE_3_STATE['H'],
    'G': ENCODE_3_STATE['H'],
    'T': ENCODE_3_STATE['C'],
    'S': ENCODE_3_STATE['C'],
    'I': ENCODE_3_STATE['C'],
    'L': ENCODE_3_STATE['C'],
}


def dssp_to_3_state(dssp: str) -> tensor:
    """Reduce eight-state DSSP to 3-state prediction.
    See: <https://www.compbio.dundee.ac.uk/jpred/references/prot_html/node17.html>
    I am using method A because it is simple.
    This is also the method used in PSIPRED:
      > The eight states (H, I, G, E, B, S, T, -) were
      > reduced to three states according to the scheme
      > outlined by Rost & Sander (1993)
    """
    return tensor([DSSP_TO_3_STATE[s] for s in dssp])


def create_windows(pssm: tensor, secondary: tensor) -> list[tuple[tensor, int]]:
    """Perform windowing, that is, splitting a sequence into 15 residue long "windows".
    7 padding residues are added to both the start and end of the sequence so that a
    prediction can be made for every residue."""
    pssm = torch.vstack([embed_pssm(pssm[:-1]), pssm[-1]])

    # pad pssm with terminus row and columns
    pssm = pad(pssm, (WINDOW_CENTRE, WINDOW_CENTRE, 0, 1))

    # set terminus indicator
    pssm[-1, :WINDOW_CENTRE] = pssm[-1, pssm.size(1) - WINDOW_CENTRE:] = torch.ones(WINDOW_CENTRE)

    # create windows
    return [(pssm[:, i:i + WINDOW_SIZE], s) for i, s in enumerate(secondary)]
