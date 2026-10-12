"""Tests for manual type overrides and integration with stats and rules."""

import pandas as pd
import pytest

from clinical_audit.inference.models import (
    ConfidenceLevel,
    InferenceResult,
    SemanticType,
)
from clinical_audit.rules.models import ConstraintType
from clinical_audit.stats.summarizer import compute_stats_summary
from app import _apply_semantic_overrides, _infer_constraint_type


def test_apply_semantic_overrides_when_empty() -> None:
    """When no overrides exist, effective inferences match original inferences."""
    base_inferences = {
        "age": InferenceResult(
            column_name="age",
            semantic_type=SemanticType.NUMERIC_CONTINUOUS,
            confidence=ConfidenceLevel.HIGH,
            score=0.95,
            rationale="Continuous numeric variable",
        ),
        "stage": InferenceResult(
            column_name="stage",
            semantic_type=SemanticType.NUMERIC_DISCRETE,
            confidence=ConfidenceLevel.MEDIUM,
            score=0.70,
            rationale="Discrete integers",
        ),
    }

    effective = _apply_semantic_overrides(base_inferences, {})
    assert effective["age"].semantic_type == SemanticType.NUMERIC_CONTINUOUS
    assert effective["stage"].semantic_type == SemanticType.NUMERIC_DISCRETE
    assert effective["stage"].rationale == "Discrete integers"


def test_apply_semantic_overrides_override_and_revert() -> None:
    """When an override is applied, it takes precedence; when cleared, it reverts."""
    base_inferences = {
        "stage": InferenceResult(
            column_name="stage",
            semantic_type=SemanticType.NUMERIC_DISCRETE,
            confidence=ConfidenceLevel.MEDIUM,
            score=0.70,
            rationale="Discrete integers",
        ),
    }

    # User overrides 'stage' to categorical_nominal
    overrides = {"stage": SemanticType.CATEGORICAL_NOMINAL}
    effective = _apply_semantic_overrides(base_inferences, overrides)

    assert effective["stage"].semantic_type == SemanticType.CATEGORICAL_NOMINAL
    assert effective["stage"].confidence == ConfidenceLevel.HIGH
    assert effective["stage"].score == 1.0
    assert "Asignado manualmente" in effective["stage"].rationale

    # User clears override (overrides is empty or key removed)
    reverted = _apply_semantic_overrides(base_inferences, {})
    assert reverted["stage"].semantic_type == SemanticType.NUMERIC_DISCRETE
    assert reverted["stage"].confidence == ConfidenceLevel.MEDIUM
    assert reverted["stage"].rationale == "Discrete integers"


def test_compute_stats_summary_with_column_types_override() -> None:
    """Overriding variable types changes which variables appear in numeric vs factor stats."""
    df = pd.DataFrame({
        "patient_id": [101, 102, 103, 104],
        "age": [45, 52, 63, 71],
        "treatment_group": [1, 2, 1, 2],
    })

    # Default without column_types: all numeric columns treated as numeric
    default_stats = compute_stats_summary(df)
    assert default_stats.n_numeric_cols == 3

    # With semantic types where patient_id is IDENTIFIER and treatment_group is BINARY
    col_types = {
        "patient_id": SemanticType.IDENTIFIER,
        "age": SemanticType.NUMERIC_CONTINUOUS,
        "treatment_group": SemanticType.BINARY,
    }

    overridden_stats = compute_stats_summary(df, column_types=col_types)
    assert overridden_stats.n_numeric_cols == 1
    assert overridden_stats.n_categorical_cols == 1
    assert [s.column for s in overridden_stats.numeric_stats] == ["age"]
    assert list(overridden_stats.freq_tables.keys()) == ["treatment_group"]


def test_infer_constraint_type_mapping() -> None:
    """Constraint type accurately maps numeric and categorical semantic types."""
    assert _infer_constraint_type(SemanticType.NUMERIC_CONTINUOUS) == ConstraintType.NUMERIC
    assert _infer_constraint_type(SemanticType.NUMERIC_DISCRETE) == ConstraintType.NUMERIC
    assert _infer_constraint_type(SemanticType.CATEGORICAL_NOMINAL) == ConstraintType.CATEGORICAL
    assert _infer_constraint_type(SemanticType.BINARY) == ConstraintType.CATEGORICAL
    assert _infer_constraint_type(SemanticType.IDENTIFIER) is None
