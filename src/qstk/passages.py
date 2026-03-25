"""Text passage extraction for Bell test experiments."""

import random
from typing import List


def prepare_passages(
    path: str,
    num_passages: int,
    passage_length: int,
) -> List[str]:
    """Extract random passages of a given character length from a text file.

    Parameters
    ----------
    path : str
        Path to the source text file.
    num_passages : int
        Number of passages to extract.
    passage_length : int
        Approximate character length of each passage.

    Returns
    -------
    list of passage strings.
    """
    with open(path, "r", encoding="utf-8") as f:
        full_text = f.read()

    full_text = " ".join(full_text.split())
    text_len = len(full_text)

    if num_passages == 0:
        return []

    passages: List[str] = []

    while len(passages) < num_passages:
        if text_len <= passage_length:
            passages.append(full_text)
            if len(passages) == 1 and num_passages > 1:
                return passages * num_passages
            break
        else:
            start_idx = random.randint(0, text_len - passage_length)
            passage = full_text[start_idx : start_idx + passage_length]
            if passage not in passages:
                passages.append(passage)

    return passages
