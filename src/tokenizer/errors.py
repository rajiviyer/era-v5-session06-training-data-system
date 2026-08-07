"""Tokenizer-related errors."""


class TokenizerFrozenError(ValueError):
    """Raised when code attempts to mutate a frozen tokenizer."""


class TokenizerLoadError(ValueError):
    """Raised when a tokenizer artifact cannot be loaded."""


class TokenizerManifestError(ValueError):
    """Raised when a tokenizer manifest is invalid."""
