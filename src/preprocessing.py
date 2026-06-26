"""Design-matrix construction and feature sets.

The same fitted preprocessing pipeline is fitted on TCGA and then applied
unchanged to CGGA, so that the external design matrix is produced by the
development-cohort encoder. Molecular markers use an explicit ``unknown``
category, reflecting pragmatic clinical use where missing molecular status is
itself informative.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- Feature sets (Section 6) -------------------------------------------------
# A: clinical core. B: clinical + canonical molecular (primary). C: integrated class.
FEATURE_SETS = {
    "A_clinical": {
        "numeric": ["age_years"],
        "categorical": ["sex", "histology_harmonised", "grade_harmonised"],
    },
    "B_clinical_molecular": {
        "numeric": ["age_years"],
        "categorical": [
            "sex",
            "histology_harmonised",
            "grade_harmonised",
            "idh_status",
            "codeletion_1p19q",
            "mgmt_promoter_methylated",
        ],
    },
    "C_integrated": {
        "numeric": ["age_years"],
        "categorical": [
            "sex",
            "integrated_class_harmonised",
            "grade_harmonised",
            "mgmt_promoter_methylated",
        ],
    },
}

PRIMARY_FEATURE_SET = "B_clinical_molecular"


@dataclass
class DesignEncoder:
    """One-hot encoder with fixed category vocabulary learned on the training set.

    Categories are pinned at fit time so that CGGA rows with categories unseen
    in TCGA map to all-zero indicator columns (out-of-domain), and the column
    space is identical across cohorts. Continuous variables are standardised
    using training-set mean/sd; missing numeric values are median-imputed using
    the training median.
    """

    numeric: list
    categorical: list
    categories_: dict = field(default_factory=dict)
    num_median_: dict = field(default_factory=dict)
    num_mean_: dict = field(default_factory=dict)
    num_std_: dict = field(default_factory=dict)
    columns_: list = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "DesignEncoder":
        for c in self.numeric:
            v = pd.to_numeric(df[c], errors="coerce")
            self.num_median_[c] = float(np.nanmedian(v))
            filled = v.fillna(self.num_median_[c])
            self.num_mean_[c] = float(filled.mean())
            self.num_std_[c] = float(filled.std(ddof=0)) or 1.0
        for c in self.categorical:
            cats = sorted(pd.Series(df[c].astype(str)).dropna().unique().tolist())
            self.categories_[c] = cats
        self.columns_ = self._build_columns()
        return self

    def _build_columns(self) -> list:
        cols = list(self.numeric)
        for c in self.categorical:
            for level in self.categories_[c]:
                cols.append(f"{c}={level}")
        return cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.numeric:
            v = pd.to_numeric(df[c], errors="coerce").fillna(self.num_median_[c])
            out[c] = (v - self.num_mean_[c]) / self.num_std_[c]
        for c in self.categorical:
            s = df[c].astype(str)
            for level in self.categories_[c]:
                out[f"{c}={level}"] = (s == level).astype(float)
        X = pd.DataFrame(out, index=df.index)
        return X.reindex(columns=self.columns_, fill_value=0.0)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def make_encoder(feature_set: str) -> DesignEncoder:
    spec = FEATURE_SETS[feature_set]
    return DesignEncoder(numeric=list(spec["numeric"]), categorical=list(spec["categorical"]))


def survival_y(df: pd.DataFrame):
    """Return (event_bool, time_days) arrays for survival fitting."""
    event = df["os_event"].astype(int).to_numpy().astype(bool)
    time = df["os_time_days"].astype(float).to_numpy()
    return event, time
