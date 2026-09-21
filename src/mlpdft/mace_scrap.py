from __future__ import annotations

from pathlib import Path

import torch
from mace.calculators.mace import MACECalculator

from mlpdft.config import MaceConfig

# ── e3nn CPU-only workaround ─────────────────────────────────────
# e3nn model pickles call torch.jit.load(buffer) without map_location,
# which initialises a CUDA context even on CPU-only machines. When CUDA
# is unavailable, patch torch.jit.load to default to CPU so models load
# without an NVIDIA driver.
if not torch.cuda.is_available():
    _jit_load_original = torch.jit.load

    def _jit_load_cpu(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return _jit_load_original(*args, **kwargs)

    torch.jit.load = _jit_load_cpu  # type: ignore[assignment]
# ──────────────────────────────────────────────────────────────────


class MaceScrap:
    """Factory for MACE calculators and raw models.

    Builds a single-model calculator from ``config.model.path`` by default,
    or a committee calculator when given multiple ``model_paths`` (which
    exposes ``forces_comm`` / ``energy_var`` for uncertainty estimation).
    """

    def __init__(self, config: MaceConfig):
        self.config: MaceConfig = config
        self._models: dict[str, torch.nn.Module] = {}

    def build_calculator(
        self,
        model_paths: str | Path | list[str | Path] | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ) -> MACECalculator:
        """Return a ``MACECalculator``.

        ``model_paths``:
          * ``None``       → use ``config.model.path`` (single model)
          * ``str``/``Path`` → single model
          * ``list``      → committee of models
        """
        if model_paths is None:
            paths = [str(self.config.model.path)]
        elif isinstance(model_paths, (str, Path)):
            paths = [str(model_paths)]
        else:
            paths = [str(p) for p in model_paths]

        return MACECalculator(
            model_paths=paths,
            device=device or self.config.device,
            default_dtype=dtype or self.config.dtype,
        )

    def build_model(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
    ) -> torch.nn.Module:
        """Load (and cache) a single raw MACE module, e.g. for inspection."""
        path = str(model_path or self.config.model.path)
        if path not in self._models:
            model = torch.load(
                path,
                map_location=device or self.config.device,
                weights_only=False,
            )
            model.eval()
            self._models[path] = model
        return self._models[path]
