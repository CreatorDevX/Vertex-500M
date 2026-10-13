"""Data pipeline package: shards, packing, download, splitting, manifests.

Backward compatible: every name that used to live in `data.py` is re-exported
here, so `from vertex500m.data import build_dataloader` (and friends) keeps
working unchanged.
"""

from .download import fetch_corpus
from .manifest import (
    INDEX_SUFFIX,
    MANIFEST_NAME,
    read_manifest,
    shard_record,
    split_shard_paths,
    validate_manifest,
    write_manifest,
)
from .packing import DocPackedDataset, index_path_for
from .shards import (
    TOKEN_BYTES,
    TOKEN_DTYPE,
    PackedTokenDataset,
    _load_shard,
    build_dataloader,
    build_dataset,
    list_shards,
    shard_num_tokens,
    synthetic_corpus,
)
from .split import plan_split

__all__ = [
    "DocPackedDataset",
    "INDEX_SUFFIX",
    "MANIFEST_NAME",
    "TOKEN_BYTES",
    "TOKEN_DTYPE",
    "PackedTokenDataset",
    "_load_shard",
    "build_dataloader",
    "build_dataset",
    "fetch_corpus",
    "index_path_for",
    "list_shards",
    "plan_split",
    "read_manifest",
    "shard_num_tokens",
    "shard_record",
    "split_shard_paths",
    "synthetic_corpus",
    "validate_manifest",
    "write_manifest",
]
