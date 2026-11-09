"""
==============================================================================
File      : src/utils/config.py
Project   : book_rec - Reducing popularity bias in book recommendation
Owner     : Amine (lead)  |  Ella (co-owner / reviewer)
------------------------------------------------------------------------------
What it does:
  Loads config/config.yaml, turns relative paths into absolute ones and sets
  the global random seed so every run is reproducible.
Input:
  - config/config.yaml (path optional, default = project root)
Output:
  - dict with all settings; seeds set for random / numpy
==============================================================================
"""
from pathlib import Path
import random

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Read the YAML config and resolve every entry of cfg['paths'] against PROJECT_ROOT."""
    path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, rel in cfg.get("paths", {}).items():
        cfg["paths"][key] = PROJECT_ROOT / rel
    return cfg


def set_seed(seed: int) -> None:
    """Set the seed for python random and numpy."""
    random.seed(seed)
    np.random.seed(seed)