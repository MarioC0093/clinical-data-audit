"""Unit tests for heuristic semantic variable type inference."""

from pathlib import Path
import pandas as pd
import pytest

from clinical_audit.inference.classifier import (
    infer_column_type,
    infer_dataset_types,
)
from clinical_audit.inference.models import (
    ConfidenceLevel,
    InferenceResult,
    SemanticType,
)
from clinical_audit.ingestion.loader import load_clinical_data


# 1. IDENTIFIERS
def test_infer_identifier_alphanumeric_high_confidence() -> None:
    """Tests alphanumeric ID pattern (e.g. P001, P002) with 100% uniqueness."""
    s = pd.Series(["P001", "P002", "P003", "P004"], name="code")
    res = infer_column_type(s)

    assert res.semantic_type == SemanticType.IDENTIFIER
    assert res.confidence == ConfidenceLevel.HIGH
    assert res.score >= 0.90
    assert "unicidad" in res.rationale.lower()


def test_infer_identifier_by_name_and_uniqueness() -> None:
    """Tests integer identifier with explicit ID naming."""
    s = pd.Series([101, 102, 103, 104], name="subject_id")
    res = infer_column_type(s)

    assert res.semantic_type == SemanticType.IDENTIFIER
    assert res.confidence == ConfidenceLevel.HIGH


def test_infer_identifier_longitudinal_medium_confidence() -> None:
    """Tests repeated patient ID in longitudinal data having MEDIUM confidence."""
    s = pd.Series(["P001", "P002", "P001", "P003", "P002", "P001"], name="patient_id")
    res = infer_column_type(s)

    assert res.semantic_type == SemanticType.IDENTIFIER
    assert res.confidence == ConfidenceLevel.MEDIUM
    assert "longitudinal" in res.rationale.lower()


# 2. BINARY VARIABLES
def test_infer_binary_standard_tokens_high_confidence() -> None:
    """Tests classic dichotomous pairs (M/F, True/False, 0/1)."""
    # M/F
    s_sex = pd.Series(["M", "F", "F", "M"], name="sex")
    res_sex = infer_column_type(s_sex)
    assert res_sex.semantic_type == SemanticType.BINARY
    assert res_sex.confidence == ConfidenceLevel.HIGH

    # 0/1 numeric
    s_flag = pd.Series([0, 1, 1, 0, 1], name="hypertension_flag")
    res_flag = infer_column_type(s_flag)
    assert res_flag.semantic_type == SemanticType.BINARY
    assert res_flag.confidence == ConfidenceLevel.HIGH


def test_infer_binary_arbitrary_strings_high_confidence() -> None:
    """Tests arbitrary 2 non-numeric categories."""
    s = pd.Series(["Tratamiento", "Control", "Control", "Tratamiento"], name="branch")
    res = infer_column_type(s)

    assert res.semantic_type == SemanticType.BINARY
    assert res.confidence == ConfidenceLevel.HIGH


def test_infer_binary_arbitrary_numeric_medium_confidence() -> None:
    """Tests numeric variable that happens to have only 2 arbitrary values."""
    s = pd.Series([50, 100, 50, 100], name="dose_mg")
    res = infer_column_type(s)

    assert res.semantic_type == SemanticType.BINARY
    assert res.confidence == ConfidenceLevel.MEDIUM


# 3. CATEGORICAL NOMINAL
def test_infer_categorical_nominal_high_confidence() -> None:
    """Tests qualitative text column with >2 categories."""
    s = pd.Series(
        ["Cardiología", "Neurología", "Urgencias", "Oncología"],
        name="department",
    )
    res = infer_column_type(s)

    assert res.semantic_type == SemanticType.CATEGORICAL_NOMINAL
    assert res.confidence == ConfidenceLevel.HIGH


# 4. NUMERIC CONTINUOUS
def test_infer_numeric_continuous_floating_decimals_high_confidence() -> None:
    """Tests continuous variables with real decimal places."""
    s = pd.Series([22.4, 31.2, 27.8, 25.1], name="bmi")
    res = infer_column_type(s)

    assert res.semantic_type == SemanticType.NUMERIC_CONTINUOUS
    assert res.confidence == ConfidenceLevel.HIGH
    assert res.evidence["has_decimals"] is True


def test_infer_numeric_continuous_biometric_integers_medium_confidence() -> None:
    """Tests clinical biometrics (e.g. blood pressure) registered as whole integers."""
    s = pd.Series([118, 145, 132, 128, 121, 110, 178], name="systolic_bp")
    res = infer_column_type(s)

    assert res.semantic_type == SemanticType.NUMERIC_CONTINUOUS
    assert res.confidence == ConfidenceLevel.MEDIUM
    assert "clínico" in res.rationale.lower()


# 5. NUMERIC DISCRETE
def test_infer_numeric_discrete_counts_high_confidence() -> None:
    """Tests count variable with name and integer values."""
    s = pd.Series([0, 1, 2, 0, 3, 1], name="num_admissions")
    res = infer_column_type(s)

    assert res.semantic_type == SemanticType.NUMERIC_DISCRETE
    assert res.confidence == ConfidenceLevel.HIGH


def test_infer_numeric_discrete_bounded_integers_medium_confidence() -> None:
    """Tests bounded non-negative integers without explicit count naming."""
    s = pd.Series([1, 2, 3, 2, 4, 1], name="stage_score")
    res = infer_column_type(s)

    assert res.semantic_type == SemanticType.NUMERIC_DISCRETE
    assert res.confidence == ConfidenceLevel.MEDIUM


# 6. AMBIGUOUS CASES (MEDIUM / LOW CONFIDENCE)
def test_infer_ambiguous_age_medium_confidence() -> None:
    """Tests age in whole integer years returning MEDIUM confidence."""
    s = pd.Series([34, 67, 52, 45, 29, 81], name="age")
    res = infer_column_type(s)

    assert res.semantic_type == SemanticType.NUMERIC_CONTINUOUS
    assert res.confidence == ConfidenceLevel.MEDIUM
    assert "edad" in res.rationale.lower()


def test_infer_ambiguous_binary_with_contaminant_low_confidence() -> None:
    """Tests binary variable corrupted by a 3rd invalid value (Yes, No, Maybe)."""
    s = pd.Series(["Yes", "No", "No", "Yes", "Maybe"], name="smoker")
    res = infer_column_type(s)

    assert res.semantic_type == SemanticType.BINARY
    assert res.confidence == ConfidenceLevel.LOW
    assert "3 categorías" in res.rationale.lower()


def test_infer_ambiguous_empty_column_low_confidence() -> None:
    """Tests completely null column returning LOW confidence."""
    s = pd.Series([None, None, None], name="empty_field")
    res = infer_column_type(s)

    assert res.confidence == ConfidenceLevel.LOW
    assert "vacía" in res.rationale.lower()


def test_infer_ambiguous_small_sample_not_misclassified_as_id() -> None:
    """Tests that a 3-row continuous metric with 100% uniqueness is NOT called an ID."""
    s = pd.Series([71.5, 68.2, 85.0], name="weight_kg")
    res = infer_column_type(s)

    assert res.semantic_type == SemanticType.NUMERIC_CONTINUOUS
    assert res.semantic_type != SemanticType.IDENTIFIER


# 7. INTEGRATION ON REAL CLINICAL DATASET
def test_infer_dataset_types_on_clinical_sample() -> None:
    """Tests classification of all variables in data/clinical_sample.csv."""
    sample_path = Path(__file__).resolve().parent.parent / "data" / "clinical_sample.csv"
    if not sample_path.exists():
        pytest.skip("data/clinical_sample.csv not found")

    df = load_clinical_data(sample_path)
    inferences = infer_dataset_types(df)

    assert len(inferences) == 6

    # patient_id -> IDENTIFIER (HIGH)
    assert inferences["patient_id"].semantic_type == SemanticType.IDENTIFIER
    assert inferences["patient_id"].confidence == ConfidenceLevel.HIGH

    # sex -> BINARY (HIGH)
    assert inferences["sex"].semantic_type == SemanticType.BINARY
    assert inferences["sex"].confidence == ConfidenceLevel.HIGH

    # bmi -> NUMERIC_CONTINUOUS (HIGH)
    assert inferences["bmi"].semantic_type == SemanticType.NUMERIC_CONTINUOUS
    assert inferences["bmi"].confidence == ConfidenceLevel.HIGH

    # systolic_bp -> NUMERIC_CONTINUOUS (MEDIUM)
    assert inferences["systolic_bp"].semantic_type == SemanticType.NUMERIC_CONTINUOUS
    assert inferences["systolic_bp"].confidence == ConfidenceLevel.MEDIUM

    # smoker -> BINARY (LOW, due to 'Maybe')
    assert inferences["smoker"].semantic_type == SemanticType.BINARY
    assert inferences["smoker"].confidence == ConfidenceLevel.LOW

    # age -> NUMERIC_CONTINUOUS (MEDIUM)
    assert inferences["age"].semantic_type == SemanticType.NUMERIC_CONTINUOUS
    assert inferences["age"].confidence == ConfidenceLevel.MEDIUM


# 8. TYPE ERRORS & SERIALIZATION
def test_infer_dataset_types_invalid_input() -> None:
    """Tests TypeError when input is not a DataFrame."""
    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        infer_dataset_types("not a dataframe")  # type: ignore


def test_inference_result_serialization() -> None:
    """Tests serialization to dictionary."""
    s = pd.Series(["A", "B"], name="col")
    res = infer_column_type(s)
    d = res.to_dict()

    assert d["column_name"] == "col"
    assert d["semantic_type"] == "binary"
    assert d["confidence"] == "HIGH"
    assert "score" in d
    assert "evidence" in d
