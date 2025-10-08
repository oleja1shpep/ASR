import re
from collections import defaultdict
from string import ascii_lowercase

import torch
from torchaudio.models.decoder import ctc_decoder

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.ctc_beam_search_decoder = ctc_decoder(
            lexicon=None,
            tokens=self.vocab,
            blank_token=self.EMPTY_TOK,
            sil_token=" ",
        )

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_beam_search(self, log_probs: torch.Tensor, lengths: torch.Tensor):
        result = self.ctc_beam_search_decoder(log_probs.exp(), lengths)
        predictions = [self.decode(hypos[0].tokens) for hypos in result]
        return predictions

    def my_ctc_beam_search(
        self, probs: torch.Tensor, beam_size: int, eps: float = 1e-10
    ) -> str:
        beams = [("", self.EMPTY_TOK, 1.0)]
        for idx, cur_step_probs in enumerate(probs):
            # expand and merge beams
            new_beams = {}

            for pref, last_char, prob in beams:
                for idx, char in self.ind2char.items():
                    char_prob = cur_step_probs[idx]
                    if char_prob < eps:
                        continue
                    cur_prob = prob * char_prob

                    if char == self.EMPTY_TOK:
                        cur_pref = pref
                    elif last_char != char:
                        cur_pref = pref + char
                    else:
                        cur_pref = pref

                    key = (cur_pref, char)
                    if key in new_beams:
                        new_beams[key] += cur_prob
                    else:
                        new_beams[key] = cur_prob

            # truncate beams
            beams = sorted(
                [
                    (pref, last_char, prob)
                    for (pref, last_char), prob in new_beams.items()
                ],
                key=lambda x: -x[2],
            )[:beam_size]

        return beams[0][0].strip()

    def ctc_decode(self, inds) -> str:
        filtered = [inds[0]]
        for i in range(1, len(inds)):
            if inds[i] != inds[i - 1]:
                filtered.append(inds[i])
        return "".join([self.ind2char[int(ind)] for ind in filtered]).strip()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
