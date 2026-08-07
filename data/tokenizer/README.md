# Tokenizer artifacts (Session 6)

Frozen **Session 2 BPE** tokenizer (decision D7), committed as `bpe_tokenizer.json`.

| File | Purpose |
|------|---------|
| `bpe_tokenizer.json` | Committed BPE artifact (~10k vocab, Metaspace) |
| `tokenizer_hash.json` | Stable `tok_*` fingerprint (vocab + merges + special tokens) |
| `tokenizer_manifest.json` | Machine-readable tokenizer manifest for shard admission |

Refresh hash and manifest sidecars from the committed artifact:

```bash
uv run python -c "import sys; from pathlib import Path; sys.path.insert(0, 'src'); from tokenizer import rebuild_bpe_tokenizer_artifact, default_tokenizer_path; root = Path('.'); rebuild_bpe_tokenizer_artifact(default_tokenizer_path(root), assignment_root=root)"
```

The legacy WordLevel `toy_tokenizer.json` was removed in P1-T03R.
