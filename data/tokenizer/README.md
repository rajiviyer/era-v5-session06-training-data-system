# Tokenizer artifacts (Session 6)

Frozen **Session 2 BPE** tokenizer (decision D7). Source: `session02/artifacts/tokenizer.json`.

| File | Purpose |
|------|---------|
| `bpe_tokenizer.json` | Committed BPE artifact (~10k vocab, Metaspace) |
| `tokenizer_hash.json` | Stable `tok_*` fingerprint (vocab + merges + special tokens) |
| `tokenizer_manifest.json` | Machine-readable tokenizer manifest for shard admission |

Refresh from Session 2 source:

```powershell
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'session06/assignment/src'); from pathlib import Path; from tokenizer import rebuild_bpe_tokenizer_artifact, default_tokenizer_path; rebuild_bpe_tokenizer_artifact(default_tokenizer_path(Path('session06/assignment')), assignment_root=Path('session06/assignment'))"
```

```bash
.venv/bin/python -c "import sys; sys.path.insert(0, 'session06/assignment/src'); from pathlib import Path; from tokenizer import rebuild_bpe_tokenizer_artifact, default_tokenizer_path; rebuild_bpe_tokenizer_artifact(default_tokenizer_path(Path('session06/assignment')), assignment_root=Path('session06/assignment'))"
```

The legacy WordLevel `toy_tokenizer.json` was removed in P1-T03R.
