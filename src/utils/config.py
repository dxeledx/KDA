from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import yaml


def load_yaml(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Union[str, os.PathLike]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
