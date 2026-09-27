"""Statistical summarizer for clinical datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ── Data containers ──────────────────────────────────────────────────────────

@dataclass
class NumericStats:
    """Descriptive statistics for a numeric variable."""

    column: str
    count: int
    missing: int
    missing_pct: float
    mean: float
    std: float
    median: float
    q1: float
    q3: float
    iqr: float
    min: float
    max: float
    skewness: float
    kurtosis: float
    cv: float  # coefficient of variation (%)


@dataclass
class CategoricalFreq:
    """Frequency table entry for a single category."""

    category: str
    count: int
    pct: float


@dataclass
class GroupedFreqRow:
    """One row in a grouped frequency table."""

    group_label: str       # value of the grouping variable
    subgroup_label: str    # value (or category label) of the second variable
    count: int
    pct: float             # % inside the group


@dataclass
class StatsSummary:
    """Complete statistical summary for a dataset."""

    # Basic dataset-level info
    n_rows: int
    n_cols: int
    n_complete_rows: int          # rows with zero missing values
    complete_row_pct: float
    n_numeric_cols: int
    n_categorical_cols: int

    # Per-variable details
    numeric_stats: List[NumericStats] = field(default_factory=list)

    # Column-level frequency tables  {col_name → list of CategoricalFreq}
    freq_tables: Dict[str, List[CategoricalFreq]] = field(default_factory=dict)

    # Grouped frequency tables  {(group_col, value_col) → list of GroupedFreqRow}
    grouped_freq: Dict[Tuple[str, str], List[GroupedFreqRow]] = field(default_factory=dict)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_factor_column(series: pd.Series, max_categories: int = 20) -> bool:
    """Returns True when a column looks like a categorical/factor variable."""
    n_unique = series.nunique(dropna=True)
    if series.dtype == object or str(series.dtype) == "category":
        return n_unique <= max_categories
    # Numeric but low-cardinality → treat as factor
    return n_unique <= max_categories and n_unique >= 2


def _is_binary_column(series: pd.Series) -> bool:
    """Returns True when a column has exactly 2 non-null distinct values."""
    return series.nunique(dropna=True) == 2


# ── Main builder ──────────────────────────────────────────────────────────────

def compute_stats_summary(df: pd.DataFrame) -> StatsSummary:
    """Computes a complete StatsSummary for *df*.

    Args:
        df: Input pandas DataFrame.

    Returns:
        StatsSummary populated with numeric stats, frequency tables, and
        grouped frequency tables for all relevant column combinations.
    """
    n_rows, n_cols = df.shape

    # Row completeness
    n_complete = int((~df.isnull().any(axis=1)).sum())
    complete_pct = round(n_complete / n_rows * 100, 1) if n_rows > 0 else 0.0

    # Identify numeric vs. categorical columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    n_numeric = len(numeric_cols)

    factor_cols = [c for c in df.columns if _is_factor_column(df[c])]
    n_categorical = len(factor_cols)

    # ── Numeric statistics ────────────────────────────────────────────────────
    numeric_stats: List[NumericStats] = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        total = len(df[col])
        missing = int(df[col].isna().sum())
        missing_pct = round(missing / total * 100, 1) if total > 0 else 0.0

        if s.empty:
            continue

        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        mean_val = float(s.mean())
        std_val = float(s.std())
        cv = round((std_val / mean_val * 100), 2) if mean_val != 0 else float("nan")

        numeric_stats.append(
            NumericStats(
                column=col,
                count=len(s),
                missing=missing,
                missing_pct=missing_pct,
                mean=round(mean_val, 4),
                std=round(std_val, 4),
                median=round(float(s.median()), 4),
                q1=round(q1, 4),
                q3=round(q3, 4),
                iqr=round(q3 - q1, 4),
                min=round(float(s.min()), 4),
                max=round(float(s.max()), 4),
                skewness=round(float(s.skew()), 4),
                kurtosis=round(float(s.kurt()), 4),
                cv=cv,
            )
        )

    # ── Frequency tables (factor columns) ────────────────────────────────────
    freq_tables: Dict[str, List[CategoricalFreq]] = {}
    for col in factor_cols:
        vc = df[col].value_counts(dropna=True)
        total_valid = vc.sum()
        freq_tables[col] = [
            CategoricalFreq(
                category=str(cat),
                count=int(cnt),
                pct=round(cnt / total_valid * 100, 1) if total_valid > 0 else 0.0,
            )
            for cat, cnt in vc.items()
        ]

    return StatsSummary(
        n_rows=n_rows,
        n_cols=n_cols,
        n_complete_rows=n_complete,
        complete_row_pct=complete_pct,
        n_numeric_cols=n_numeric,
        n_categorical_cols=n_categorical,
        numeric_stats=numeric_stats,
        freq_tables=freq_tables,
        grouped_freq={},  # computed on-demand in the UI
    )


def compute_grouped_freq(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> List[GroupedFreqRow]:
    """Computes a grouped frequency table for two categorical variables.

    For binary *value_col*: returns the proportion of the "positive" category
    (the non-first category alphabetically, or the larger numeric value) inside
    each group of *group_col*.

    For non-binary *value_col*: returns raw counts and percentages of each
    category inside each group.

    Args:
        df: Input DataFrame.
        group_col: Column to group by (rows).
        value_col: Column whose distribution to summarise within each group.

    Returns:
        List of GroupedFreqRow objects ready for display.
    """
    rows: List[GroupedFreqRow] = []
    is_binary = _is_binary_column(df[value_col])

    grouped = df.dropna(subset=[group_col, value_col]).groupby(group_col, observed=True)

    for group_val, grp_df in grouped:
        grp_size = len(grp_df)
        vc = grp_df[value_col].value_counts(dropna=True)

        if is_binary:
            # Show proportion of the "positive" / second category
            cats_sorted = sorted(vc.index.tolist(), key=lambda x: str(x))
            pos_cat = cats_sorted[-1]  # last alphabetically / numerically
            pos_count = int(vc.get(pos_cat, 0))
            rows.append(
                GroupedFreqRow(
                    group_label=str(group_val),
                    subgroup_label=f"prop({pos_cat})",
                    count=pos_count,
                    pct=round(pos_count / grp_size * 100, 1) if grp_size > 0 else 0.0,
                )
            )
        else:
            for cat, cnt in vc.items():
                rows.append(
                    GroupedFreqRow(
                        group_label=str(group_val),
                        subgroup_label=str(cat),
                        count=int(cnt),
                        pct=round(cnt / grp_size * 100, 1) if grp_size > 0 else 0.0,
                    )
                )

    return rows
