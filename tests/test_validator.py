"""Unit tests for configurable rules evaluation engine."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clinical_audit.ingestion.loader import load_clinical_data
from clinical_audit.rules.config import load_rules_from_yaml
from clinical_audit.rules.models import (
    CategoricalRuleConfig,
    ColumnRuleConfig,
    ConstraintType,
    NumericRuleConfig,
    RulesConfig,
    ValidationReport,
)
from clinical_audit.rules.validator import validate_dataframe


def test_validator_clean_dataset() -> None:
    """Tests that a fully compliant DataFrame produces zero violations."""
    df = pd.DataFrame(
        {
            "age": [25, 45, 65],
            "sex": ["F", "M", "F"],
            "bmi": [21.5, 24.2, 28.0],
        }
    )
    config = RulesConfig(
        columns={
            "age": ColumnRuleConfig(
                column_name="age",
                rule_type=ConstraintType.NUMERIC,
                numeric=NumericRuleConfig(min_value=0, max_value=120),
            ),
            "sex": ColumnRuleConfig(
                column_name="sex",
                rule_type=ConstraintType.CATEGORICAL,
                categorical=CategoricalRuleConfig(allowed=["F", "M"]),
            ),
            "bmi": ColumnRuleConfig(
                column_name="bmi",
                rule_type=ConstraintType.NUMERIC,
                numeric=NumericRuleConfig(min_value=10, max_value=80),
            ),
        }
    )

    report = validate_dataframe(df, config)
    assert isinstance(report, ValidationReport)
    assert report.is_valid
    assert report.total_violations == 0
    assert report.violations == []


def test_validator_does_not_confuse_missing_with_invalid() -> None:
    """Verifies that missing values (NaN/None) are NOT reported as invalid values."""
    df = pd.DataFrame(
        {
            "bmi": [22.0, np.nan, None, 25.5],
            "sex": ["F", None, np.nan, "M"],
        }
    )
    config = RulesConfig(
        columns={
            "bmi": ColumnRuleConfig(
                column_name="bmi",
                rule_type=ConstraintType.NUMERIC,
                numeric=NumericRuleConfig(min_value=10, max_value=80),
            ),
            "sex": ColumnRuleConfig(
                column_name="sex",
                rule_type=ConstraintType.CATEGORICAL,
                categorical=CategoricalRuleConfig(allowed=["F", "M"]),
            ),
        }
    )

    report = validate_dataframe(df, config)
    # The nulls in bmi and sex must NOT trigger violations
    assert report.is_valid
    assert report.total_violations == 0
    assert report.violations_by_column["bmi"] == 0
    assert report.violations_by_column["sex"] == 0


def test_validator_detects_numeric_range_violations() -> None:
    """Tests detection of values exceeding min or max thresholds."""
    df = pd.DataFrame(
        {
            "age": [30, -5, 40, 999],
        }
    )
    config = RulesConfig(
        columns={
            "age": ColumnRuleConfig(
                column_name="age",
                rule_type=ConstraintType.NUMERIC,
                numeric=NumericRuleConfig(min_value=0, max_value=120),
            ),
        }
    )

    report = validate_dataframe(df, config)
    assert not report.is_valid
    assert report.total_violations == 2
    assert report.violations_by_column["age"] == 2

    v_min = report.violations[0]
    assert v_min.column == "age"
    assert v_min.row_index == 1
    assert v_min.invalid_value == -5
    assert v_min.rule_type == "numeric_min"

    v_max = report.violations[1]
    assert v_max.column == "age"
    assert v_max.row_index == 3
    assert v_max.invalid_value == 999
    assert v_max.rule_type == "numeric_max"


def test_validator_detects_categorical_violations() -> None:
    """Tests detection of categories outside the allowed set."""
    df = pd.DataFrame(
        {
            "smoker": ["Yes", "No", "Maybe", "Unknown"],
        }
    )
    config = RulesConfig(
        columns={
            "smoker": ColumnRuleConfig(
                column_name="smoker",
                rule_type=ConstraintType.CATEGORICAL,
                categorical=CategoricalRuleConfig(allowed=["Yes", "No"]),
            ),
        }
    )

    report = validate_dataframe(df, config)
    assert not report.is_valid
    assert report.total_violations == 2

    v1, v2 = report.violations
    assert v1.column == "smoker"
    assert v1.row_index == 2
    assert v1.invalid_value == "Maybe"
    assert v1.rule_type == "categorical_allowed"

    assert v2.column == "smoker"
    assert v2.row_index == 3
    assert v2.invalid_value == "Unknown"


def test_validator_detects_type_mismatch_in_numeric() -> None:
    """Tests non-numeric string values appearing in a numeric column."""
    df = pd.DataFrame(
        {
            "systolic_bp": [120, "one-hundred", 135],
        }
    )
    config = RulesConfig(
        columns={
            "systolic_bp": ColumnRuleConfig(
                column_name="systolic_bp",
                rule_type=ConstraintType.NUMERIC,
                numeric=NumericRuleConfig(min_value=50, max_value=300),
            ),
        }
    )

    report = validate_dataframe(df, config)
    assert report.total_violations == 1
    assert report.violations[0].rule_type == "numeric_type"
    assert report.violations[0].invalid_value == "one-hundred"


def test_validator_missing_column_handled() -> None:
    """Tests that configured columns missing from the dataset are recorded."""
    df = pd.DataFrame({"age": [30, 40]})
    config = RulesConfig(
        columns={
            "cholesterol": ColumnRuleConfig(
                column_name="cholesterol",
                rule_type=ConstraintType.NUMERIC,
                numeric=NumericRuleConfig(min_value=50, max_value=400),
            ),
        }
    )

    report = validate_dataframe(df, config)
    assert report.total_violations == 1
    assert report.violations[0].rule_type == "missing_column"


def test_validation_on_actual_clinical_sample() -> None:
    """Integration test executing YAML rules against data/clinical_sample.csv."""
    _ROOT = Path(__file__).resolve().parent.parent
    data_path = _ROOT / "data" / "clinical_sample.csv"
    config_path = _ROOT / "config" / "clinical_rules.yaml"

    if not data_path.exists() or not config_path.exists():
        pytest.skip("Sample dataset or config not found")

    df = load_clinical_data(data_path)
    config = load_rules_from_yaml(config_path)

    report = validate_dataframe(df, config)

    # In the modified clinical_sample.csv:
    # 1. P004 (index 3) has age=999 -> violates max: 120
    # 2. P010 (index 9) has smoker='Maybe' -> violates allowed: ['Yes', 'No']
    # 3. P005 and P006 have bmi=NaN -> must NOT violate numeric rules
    assert report.total_records == 10
    assert report.total_violations == 2
    assert not report.is_valid

    # Exact violations verification
    violations_summary = {
        (v.column, v.row_index, str(v.invalid_value)): v.rule_type
        for v in report.violations
    }

    assert ("age", 3, "999") in violations_summary
    assert violations_summary[("age", 3, "999")] == "numeric_max"

    assert ("smoker", 9, "Maybe") in violations_summary
    assert violations_summary[("smoker", 9, "Maybe")] == "categorical_allowed"

    # Ensure BMI has zero violations despite missing values
    assert report.violations_by_column.get("bmi", 0) == 0


def test_validation_report_serialization() -> None:
    """Tests serialization of ValidationReport to dictionary."""
    df = pd.DataFrame({"age": [999]})
    config = RulesConfig(
        columns={
            "age": ColumnRuleConfig(
                column_name="age",
                rule_type=ConstraintType.NUMERIC,
                numeric=NumericRuleConfig(min_value=0, max_value=120),
            )
        }
    )
    report = validate_dataframe(df, config)
    data = report.to_dict()

    assert data["total_records"] == 1
    assert data["total_violations"] == 1
    assert data["is_valid"] is False
    assert len(data["violations"]) == 1
    assert data["violations"][0]["invalid_value"] == 999
