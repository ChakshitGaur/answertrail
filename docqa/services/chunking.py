import re

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_long(piece: str, size: int) -> list[str]:
    """Break a piece longer than `size` (e.g. a code block) at spaces."""
    parts = []
    while len(piece) > size:
        cut = piece.rfind(" ", size // 2, size)
        cut = cut if cut != -1 else size
        parts.append(piece[:cut].strip())
        piece = piece[cut:].strip()
    if piece:
        parts.append(piece)
    return parts


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Pack whole sentences into chunks of at most `size` characters.

    Consecutive chunks share trailing sentences totalling at most `overlap` characters, so a
    definition is not split from its lead-in. Sentences longer than `size` are split at spaces.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")

    text = " ".join(text.split())
    if not text:
        return []

    units = []
    for sentence in _SENTENCE_END.split(text):
        units.extend(_split_long(sentence, size) if len(sentence) > size else [sentence])

    chunks: list[str] = []
    current: list[str] = []
    length = 0  # characters in `current`, joined by single spaces
    for unit in units:
        added = len(unit) + (1 if current else 0)
        if current and length + added > size:
            chunks.append(" ".join(current))
            tail, tail_len = [], 0
            for prev in reversed(current):
                if tail_len + len(prev) + (1 if tail else 0) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + (1 if len(tail) > 1 else 0)
            current, length = tail, tail_len
            added = len(unit) + (1 if current else 0)
        current.append(unit)
        length += added
    if current:
        chunks.append(" ".join(current))

    # drop a trailing chunk that only repeats the previous chunk's overlap
    if len(chunks) > 1 and chunks[-1] in chunks[-2]:
        chunks.pop()
    return chunks
