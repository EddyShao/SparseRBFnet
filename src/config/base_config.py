# config/base_config.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping
from copy import deepcopy
import pathlib
import yaml  # requires pyyaml


# ---------------------------------------------------------------
# Config class (nested, dot-access, project-agnostic)
# ---------------------------------------------------------------

@dataclass
class Config:
    """
    A simple container for nested config data with dot-access.
    Internally stores everything in `self.data`.
    """

    data: Dict[str, Any] = field(default_factory=dict)

    # --- dict-style access ---
    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    # --- dot-style access ---
    def __getattr__(self, key):
    # Called only if normal attribute lookup fails
        try:
            return self.data[key]
        except KeyError:
            # Very important: raise AttributeError so hasattr(), deepcopy(), etc. behave
            raise AttributeError(key)

    def __setattr__(self, key: str, value: Any):
        if key == "data":
            super().__setattr__(key, value)
        else:
            self.data[key] = value

    # Convert back to a pure dict (for saving)
    def to_dict(self) -> Dict[str, Any]:
        """
        Return a plain Python dict corresponding to this config,
        recursively unwrapping any nested Config objects.
        """
        def unwrap(obj):
            if isinstance(obj, Config):
                # unwrap its internal dict
                return {k: unwrap(v) for k, v in obj.data.items()}
            elif isinstance(obj, dict):
                return {k: unwrap(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [unwrap(v) for v in obj]
            else:
                return obj

        return unwrap(self.data)

    def __repr__(self):
        return f"Config({self.data})"


# ---------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------

def _to_nested_config(obj: Any) -> Any:
    """
    Recursively turn dicts → Config objects, lists → list of Configs.
    """
    if isinstance(obj, Mapping):
        return Config({k: _to_nested_config(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_nested_config(v) for v in obj]
    else:
        return obj


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dicts.
    Values in `override` always win.
    """
    result = deepcopy(base)
    for k, v in override.items():
        if (k in result
                and isinstance(result[k], Mapping)
                and isinstance(v, Mapping)):
            result[k] = _merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------
# YAML + CLI merging
# ---------------------------------------------------------------

def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load YAML file into a Python dictionary.
    If the file is empty, returns {}.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML config file does not exist: {path}")

    with path.open("r") as f:
        loaded = yaml.safe_load(f)

    return loaded if loaded is not None else {}


def load_config_from_args(args, yaml_key: str = "config") -> Config:
    """
    Load configuration from YAML + CLI arguments (argparse).

    Rules:
      1. If args.config exists, load YAML as base config.
      2. CLI args override YAML for all non-None fields.
      3. Return as nested Config object with dot-access.

    Example:
      cfg = load_config_from_args(args)
      cfg.solver.alpha
      cfg.pde.dim
    """
    args_dict = vars(args)
    yaml_path = args_dict.get(yaml_key, None)

    if yaml_path:
        base_cfg = load_yaml_config(yaml_path)
        cli_override = {k: v for k, v in args_dict.items() if v is not None}
        merged = _merge_dicts(base_cfg, cli_override)
    else:
        merged = args_dict

    return _to_nested_config(merged)