from __future__ import annotations

from typing import Any, Mapping, Optional

try:
    import optuna
except ImportError:
    optuna = None  # type: ignore

def apply_optuna_hyperparameter_sweep(cfg: Any, trial: Any, *, sweep_section: Optional[Mapping[str, Any]] = None) -> None:
    """
    Read sweep definitions from config and apply suggested values via ``cfg.set``.
    Each entry under ``hyperparameter_sweep`` should look like::
        optimizer/lr:
    """
    if optuna is None:
        raise ImportError("optuna is required for apply_optuna_hyperparameter_sweep")

    sweep_config: Mapping[str, Any] = (
        sweep_section if sweep_section is not None else cfg.get("hyperparameter_sweep") or {}
    )
    if not sweep_config:
        return

    for raw_key, settings in sweep_config.items():
        if not isinstance(settings, Mapping):
            continue
        method_name = settings.get("method")
        if not method_name:
            raise ValueError(f"hyperparameter_sweep entry {raw_key!r} missing 'method'")
        params = dict(settings.get("params") or {})
        cfg_path = str(raw_key).replace("/", ".")
        suggest = getattr(trial, str(method_name))
        suggested_value = suggest(**params)
        cfg.set(cfg_path, suggested_value)
