"""vertex500m: a dense 503.8M reproduction of the DeepSeek-V4.1-Flash architecture.

The architecture of record is `DeepSeek-V4.1-Flash_Architecture_Spec.md`; the
MoE -> dense conversion is `vertex500m-dense.md`; the hyper-connection variant study is
`mHC-lite.md`. See README.md for what deviates and why.
"""

from .config import PRESET, MOE_PRESET, Cfg

__all__ = ["PRESET", "MOE_PRESET", "Cfg"]
