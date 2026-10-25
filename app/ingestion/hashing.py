import hashlib
from pathlib import Path


def hash_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hex digest of a file, reading it in blocks."""
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(block_size):
            sha.update(block)
    return sha.hexdigest()