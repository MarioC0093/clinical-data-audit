"""Unit tests for dataset profiling and quality reporting."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clinical_audit.core.models import QualityReport
from clinical_audit.ingestion.loader import load_clinical_data
from clinical_audit.quality.profiler import generate_quality_report


def test_generate_quality_report_basic(sample_clinical_df: pd.DataFrame) -> None:
    """Tests basic profiling metrics on a standard valid DataFrame."""
    report = generate_quality_report(sample_clinical_df, file_name="mock.csv")

    assert isinstance(report, QualityReport)
    assert report.file_name == "mock.csv"
    assert report.row_count == 3
    assert report.column_count == 4
    assert report.exact_duplicates_count == 0

    assert set(report.columns.keys()) == {
        "patient_id",
        "age",
        "gender",
        "diagnosis_code",
    }

    age_col = report.columns["age"]
    assert age_col.name == "age"
    assert "int" in age_col.dtype
    assert age_col.missing_count == 0
    assert age_col.missing_percentage == 0.0
    assert age_col.unique_count == 3


def test_generate_quality_report_exact_duplicates() -> None:
    """Tests exact duplicate row detection."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 3, 2],
            "val": ["A", "B", "B", "C", "B"],
        }
    )
    report = generate_quality_report(df)

    assert report.row_count == 5
    # Row (2, 'B') appears 3 times, so there are 2 duplicate rows beyond first occurrence
    assert report.exact_duplicates_count == 2


def test_generate_quality_report_missing_values() -> None:
    """Tests missing count and percentage calculations."""
    df = pd.DataFrame(
        {
            "numeric": [10.0, np.nan, 30.0, np.nan],
            "text": ["Alpha", None, "Gamma", "Delta"],
        }
    )
    report = generate_quality_report(df)

    num_col = report.columns["numeric"]
    assert num_col.missing_count == 2
    assert num_col.missing_percentage == 50.0
    assert num_col.unique_count == 2

    text_col = report.columns["text"]
    assert text_col.missing_count == 1
    assert text_col.missing_percentage == 25.0
    assert text_col.unique_count == 3


def test_generate_quality_report_empty_dataframe() -> None:
    """Tests handling of a completely empty DataFrame."""
    df = pd.DataFrame()
    report = generate_quality_report(df)

    assert report.row_count == 0
    assert report.column_count == 0
    assert report.exact_duplicates_count == 0
    assert report.columns == {}


def test_generate_quality_report_empty_with_columns() -> None:
    """Tests DataFrame with column definitions but zero rows."""
    df = pd.DataFrame(columns=["col_a", "col_b"])
    report = generate_quality_report(df)

    assert report.row_count == 0
    assert report.column_count == 2
    assert report.exact_duplicates_count == 0
    assert "col_a" in report.columns
    assert report.columns["col_a"].missing_count == 0
    assert report.columns["col_a"].missing_percentage == 0.0
    assert report.columns["col_a"].unique_count == 0


def test_generate_quality_report_invalid_type() -> None:
    """Tests that TypeError is raised when input is not a DataFrame."""
    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        generate_quality_report("not a dataframe")  # type: ignore

    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        generate_quality_report([1, 2, 3])  # type: ignore


def test_quality_report_serialization(sample_clinical_df: pd.DataFrame) -> None:
    """Tests serialization of QualityReport to dictionary."""
    report = generate_quality_report(sample_clinical_df, file_name="sample.csv")
    data = report.to_dict()

    assert data["file_name"] == "sample.csv"
    assert data["row_count"] == 3
    assert data["column_count"] == 4
    assert data["exact_duplicates_count"] == 0
    assert "age" in data["columns"]
    assert data["columns"]["age"]["missing_count"] == 0
    assert "created_at" in data


def test_generate_quality_report_on_actual_clinical_sample() -> None:
    """Tests profiling against the actual data/clinical_sample.csv file."""
    sample_file = Path(__file__).resolve().parent.parent / "data" / "clinical_sample.csv"
    if not sample_file.exists():
        pytest.skip("data/clinical_sample.csv not present in workspace")

    df = load_clinical_data(sample_file)
    report = generate_quality_report(df, file_name="clinical_sample.csv")

    assert report.row_count == 10
    assert report.column_count == 6
    assert report.exact_duplicates_count == 0

    # Test specific known properties of clinical_sample.csv
    assert "bmi" in report.columns
    assert report.columns["bmi"].missing_count == 2
    assert report.columns["bmi"].missing_percentage == 20.0
    assert report.columns["bmi"].unique_count == 8

    assert "age" in report.columns
    assert report.columns["age"].missing_count == 0
    assert report.columns["age"].missing_percentage == 0.0
    assert report.columns["age"].unique_count == 9  # 45 is repeated

    assert "sex" in report.columns
    assert report.columns["sex"].unique_count == 2

    assert "patient_id" in report.columns
    assert report.columns["patient_id"].unique_count == 10
