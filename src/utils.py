"""Shared utilities: paths, logging, constants, reproducibility.

This module centralises configuration so that every script resolves the same
directory layout and uses the same random seed and survival-time conventions.
"""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- Reproducibility -------------------------------------------------------
GLOBAL_SEED = 20240626

# --- Survival-time conventions ---------------------------------------------
# Months are converted to days using the mean Gregorian month length. This is
# applied only to TCGA (recorded in months); CGGA records overall survival in
# days and is therefore left unchanged.
DAYS_PER_MONTH = 30.4375

# Landmark prediction horizons (days). The primary horizon is 24 months.
HORIZONS_MONTHS = (12, 24, 36)
PRIMARY_HORIZON_MONTHS = 24
HORIZONS_DAYS = tuple(int(round(m * DAYS_PER_MONTH)) for m in HORIZONS_MONTHS)
PRIMARY_HORIZON_DAYS = int(round(PRIMARY_HORIZON_MONTHS * DAYS_PER_MONTH))

# Monthly prediction grid (1-36 months) used for the integrated Brier score and
# for all stored survival predictions; landmark horizons are columns of it.
GRID_MONTHS = tuple(range(1, 37))
GRID_DAYS = tuple(float(round(m * DAYS_PER_MONTH)) for m in GRID_MONTHS)
HORIZON_GRID_IDX = tuple(GRID_MONTHS.index(m) for m in HORIZONS_MONTHS)
PRIMARY_GRID_IDX = GRID_MONTHS.index(PRIMARY_HORIZON_MONTHS)

# CGGA external-validation releases used in the analysis.
CGGA_PRIMARY_RELEASES = ("mRNAseq_693", "mRNAseq_325")
CGGA_SENSITIVITY_RELEASES = ("mRNA_array_301", "WEseq_286")
CGGA_ALL_RELEASES = CGGA_PRIMARY_RELEASES + CGGA_SENSITIVITY_RELEASES


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data_raw(self) -> Path:
        return self.root / "data_raw"

    @property
    def tcga_raw(self) -> Path:
        return self.data_raw / "tcga" / "lgggbm_tcga_pub"

    @property
    def cgga_raw(self) -> Path:
        return self.data_raw / "cgga"

    @property
    def data_processed(self) -> Path:
        return self.root / "data_processed"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"

    @property
    def tables(self) -> Path:
        return self.outputs / "tables"

    @property
    def models(self) -> Path:
        return self.outputs / "models"

    @property
    def logs(self) -> Path:
        return self.outputs / "logs"

    @property
    def checksums(self) -> Path:
        return self.outputs / "checksums"

    def ensure(self) -> "Paths":
        for p in (
            self.data_processed,
            self.figures,
            self.tables,
            self.models,
            self.logs,
            self.checksums,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths() -> Paths:
    """Resolve the repository root from this file's location."""
    root = Path(__file__).resolve().parents[1]
    return Paths(root=root).ensure()


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
