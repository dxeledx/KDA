#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import BCIDataLoader
from src.utils.config import load_yaml
from src.utils.logger import get_logger


logger = get_logger("preprocess_all")


def main() -> None:
    data_cfg = load_yaml("configs/data_config.yaml")
    loader = BCIDataLoader.from_config(data_cfg)

    subjects = list(data_cfg["dataset"]["subjects"])
    logger.info("Caching processed data for %s subjects...", len(subjects))
    for sid in subjects:
        loader.ensure_subject_cached(int(sid))
    logger.info("Done. Cached files in %s", loader.processed_dir)


if __name__ == "__main__":
    main()
