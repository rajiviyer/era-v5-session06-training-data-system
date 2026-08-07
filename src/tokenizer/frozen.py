"""Frozen Hugging Face tokenizer wrapper for Session 6."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

from .bpe import BPE_ARTIFACT_NAME, ensure_bpe_tokenizer_artifact
from .errors import TokenizerFrozenError, TokenizerLoadError
from .hash import compute_tokenizer_hash_from_artifact, persist_tokenizer_hash
from .manifest import write_tokenizer_manifest

UNK = "<unk>"


class FrozenTokenizer:
    """Load-once tokenizer wrapper; runtime mutation is rejected."""

    __slots__ = ("_artifact_path", "_locked", "_tokenizer")

    def __init__(self, artifact_path: Path, tokenizer: Tokenizer) -> None:
        object.__setattr__(self, "_artifact_path", artifact_path.resolve())
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "_locked", True)

    @classmethod
    def from_file(cls, path: Path) -> FrozenTokenizer:
        """Load a tokenizer JSON artifact and freeze it."""
        artifact_path = path.resolve()
        if not artifact_path.is_file():
            raise TokenizerLoadError(f"tokenizer artifact not found: {artifact_path}")
        try:
            tokenizer = Tokenizer.from_file(str(artifact_path))
        except Exception as exc:  # noqa: BLE001 - surface library load failures
            raise TokenizerLoadError(
                f"failed to load tokenizer artifact: {artifact_path}"
            ) from exc
        return cls(artifact_path, tokenizer)

    @classmethod
    def load_default(cls, assignment_root: Path) -> FrozenTokenizer:
        """Load the committed Session 2 BPE tokenizer under data/tokenizer/."""
        root = assignment_root.resolve()
        path = ensure_bpe_tokenizer_artifact(
            root / "data" / "tokenizer" / BPE_ARTIFACT_NAME,
            assignment_root=root,
        )
        persist_tokenizer_hash(path)
        write_tokenizer_manifest(path)
        return cls.from_file(path)

    @property
    def tokenizer_hash(self) -> str:
        """Stable hash fingerprinting vocab, merges, and special tokens."""
        return compute_tokenizer_hash_from_artifact(self.artifact_path)

    @property
    def artifact_path(self) -> Path:
        return self._artifact_path

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def unk_token(self) -> str:
        return UNK

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token IDs. Same input always yields the same IDs."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        """Decode token IDs back to text."""
        if not isinstance(token_ids, list):
            raise TypeError("token_ids must be a list")
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_vocab(self) -> dict[str, int]:
        """Return a copy of the vocabulary mapping."""
        return dict(self._tokenizer.get_vocab())

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_locked", False):
            raise TokenizerFrozenError(f"cannot mutate frozen tokenizer field '{name}'")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise TokenizerFrozenError(f"cannot delete frozen tokenizer field '{name}'")
