import re
from collections import defaultdict
from string import ascii_lowercase

import torch

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

    def expand_and_merge_beams(
        self, dp: dict[tuple[str, str], float], cur_step_probs: torch.Tensor
    ) -> dict[tuple[str, str], float]:
        new_dp = defaultdict(float)

        for (pref, last_char), prob in dp.items():
            for idx, char in self.ind2char.items():
                cur_prob = prob * cur_step_probs[idx]

                if char == self.EMPTY_TOK:
                    cur_pref = pref
                elif last_char != char:
                    cur_pref = pref + char
                else:
                    cur_pref = pref

                new_dp[(cur_pref, char)] += cur_prob
        return new_dp

    def truncate_beams(
        self, dp: dict[tuple[str, str], float], beam_size: int
    ) -> dict[tuple[str, str], float]:
        return dict(sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size])

    def my_ctc_beam_search(
        self, probs: torch.Tensor, length: int, beam_size: int
    ) -> str:
        dp = {
            ("", self.EMPTY_TOK): 1.0,
        }
        for idx in range(length):
            cur_step_probs = probs[idx]
            dp = self.expand_and_merge_beams(dp, cur_step_probs)
            dp = self.truncate_beams(dp, beam_size)
        (pref, _), _ = max(list(dp.items()), key=lambda x: x[1])
        return pref.strip()

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
