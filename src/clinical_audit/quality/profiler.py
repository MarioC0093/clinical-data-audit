"""Dataset profiling and basic data quality inspection."""

from typing import Optional
import pandas as pd

from clinical_audit.core.models import ColumnQualitySummary, QualityReport


def generate_quality_report(
    df: pd.DataFrame,
    file_name: Optional[str] = None,
) -> QualityReport:
    """Generates a structured quality report for a clinical DataFrame.

    Calculates high-level structural metrics and per-variable quality indicators:
    - Total number of rows and columns.
    - Number of exact duplicate rows.
    - Number of missing values per variable.
    - Percentage of missing values per variable.
    - Number of unique (distinct non-null) values per variable.
    - Detected pandas data type per variable.

    Args:
        df: The pandas DataFrame to analyze.
        file_name: Optional label or filename associated with the dataset.

    Returns:
        QualityReport: A structured object containing the dataset and column metrics.

    Raises:
        TypeError: If `df` is not an instance of `pandas.DataFrame`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, but received {type(df).__name__}."
        )

    row_count = len(df)
    column_count = len(df.columns)
    exact_duplicates = int(df.duplicated().sum()) if row_count > 0 else 0

    columns_summary = {}
    for col_name in df.columns:
        series = df[col_name]
        col_str_name = str(col_name)
        missing_count = int(series.isna().sum())
        missing_percentage = (
            round((missing_count / row_count) * 100.0, 2) if row_count > 0 else 0.0
        )
        unique_count = int(series.nunique(dropna=True))
        dtype_str = str(series.dtype)

        columns_summary[col_str_name] = ColumnQualitySummary(
            name=col_str_name,
            dtype=dtype_str,
            missing_count=missing_count,
            missing_percentage=missing_percentage,
            unique_count=unique_count,
        )

    return QualityReport(
        row_count=row_count,
        column_count=column_count,
        exact_duplicates_count=exact_duplicates,
        columns=columns_summary,
        file_name=file_name,
    )
