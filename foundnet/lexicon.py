import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

import k2
import torch


def read_lexicon(filename: str) -> List[Tuple[str, List[str]]]:
    """Read a lexicon from `filename`.

    Each line in the lexicon contains "word p1 p2 p3 ...".
    That is, the first field is a word and the remaining
    fields are tokens. Fields are separated by space(s).

    Args:
      filename:
        Path to the lexicon.txt

    Returns:
      A list of tuples., e.g., [('w', ['p1', 'p2']), ('w1', ['p3, 'p4'])]
    """
    ans = []

    with open(filename, "r", encoding="utf-8") as f:
        whitespace = re.compile("[ \t]+")
        for line in f:
            a = whitespace.split(line.strip(" \t\r\n"))
            if len(a) == 0:
                continue

            if len(a) < 2:
                logging.info(f"Found bad line {line} in lexicon file {filename}")
                logging.info("Every line is expected to contain at least 2 fields")
                sys.exit(1)
            word = a[0]
            if word == "<eps>":
                logging.info(f"Found bad line {line} in lexicon file {filename}")
                logging.info("<eps> should not be a valid word")
                sys.exit(1)

            tokens = a[1:]
            ans.append((word, tokens))

    return ans


def write_lexicon(filename: str, lexicon: List[Tuple[str, List[str]]]) -> None:
    """Write a lexicon to a file.

    Args:
      filename:
        Path to the lexicon file to be generated.
      lexicon:
        It can be the return value of :func:`read_lexicon`.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for word, tokens in lexicon:
            f.write(f"{word} {' '.join(tokens)}\n")

class Lexicon(object):
    """Phone based lexicon."""

    def __init__(
        self,
        lang_dir: Path,
        disambig_pattern: str = re.compile(r"^#\d+$"),
    ):
        """
        Args:
          lang_dir:
            Path to the lang directory. It is expected to contain the following
            files:
                - tokens.txt
                - words.txt
                - L.pt
            The above files are produced by the script `prepare.sh`. You
            should have run that before running the training code.
          disambig_pattern:
            It contains the pattern for disambiguation symbols.
        """
        lang_dir = Path(lang_dir)
        self.token_table = k2.SymbolTable.from_file(lang_dir / "tokens.txt")
        self.word_table = k2.SymbolTable.from_file(lang_dir / "words.txt")

        if (lang_dir / "Linv.pt").exists():
            logging.info(f"Loading pre-compiled {lang_dir}/Linv.pt")
            L_inv = k2.Fsa.from_dict(torch.load(lang_dir / "Linv.pt"))
        else:
            logging.info("Converting L.pt to Linv.pt")
            L = k2.Fsa.from_dict(torch.load(lang_dir / "L.pt"))
            L_inv = k2.arc_sort(L.invert())
            torch.save(L_inv.as_dict(), lang_dir / "Linv.pt")

        # We save L_inv instead of L because it will be used to intersect with
        # transcript FSAs, both of whose labels are word IDs.
        self.L_inv = L_inv
        self.disambig_pattern = disambig_pattern

    @property
    def tokens(self) -> List[int]:
        """Return a list of token IDs excluding those from
        disambiguation symbols.

        Caution:
          0 is not a token ID so it is excluded from the return value.
        """
        symbols = self.token_table.symbols
        ans = []
        for s in symbols:
            if not self.disambig_pattern.match(s):
                ans.append(self.token_table[s])
        if 0 in ans:
            ans.remove(0)
        ans.sort()
        return ans

