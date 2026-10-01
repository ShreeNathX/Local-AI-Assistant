"""
slm.tokenizer.constants — Special tokens, byte mappings, and regex patterns for BBPE.
"""

from __future__ import annotations

import regex as re
from typing import Dict, List

# ---------------------------------------------------------------------------
# Special Tokens Definition
# ---------------------------------------------------------------------------

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
MASK_TOKEN = "<mask>"

# Chat and Tool calling delimiters
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
TOOL_CALL_TOKEN = "<|tool_call|>"
TOOL_RESULT_TOKEN = "<|tool_result|>"
THOUGHT_TOKEN = "<|thought|>"

SPECIAL_TOKENS: List[str] = [
    PAD_TOKEN,           # 0
    UNK_TOKEN,           # 1
    BOS_TOKEN,           # 2
    EOS_TOKEN,           # 3
    MASK_TOKEN,          # 4
    IM_START_TOKEN,      # 5
    IM_END_TOKEN,        # 6
    TOOL_CALL_TOKEN,     # 7
    TOOL_RESULT_TOKEN,   # 8
    THOUGHT_TOKEN,       # 9
]

SPECIAL_TOKEN_IDS: Dict[str, int] = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)

# ---------------------------------------------------------------------------
# Pre-tokenization Regex (GPT-4 / Llama 3 style)
# Handles contractions, alphabetic runs with leading space, numbers, symbols, spaces
# ---------------------------------------------------------------------------

PRE_TOKENIZE_REGEX = re.compile(
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
)


def bytes_to_unicode() -> Dict[int, str]:
    """
    Returns a bidirectional mapping between every byte (0..255) and a visible
    printable Unicode character, avoiding control characters and invisible spaces.
    Identical to GPT-2 byte encoding to ensure full invertible roundtripping.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def unicode_to_bytes() -> Dict[str, int]:
    """Inverse of bytes_to_unicode."""
    b2u = bytes_to_unicode()
    return {v: k for k, v in b2u.items()}
