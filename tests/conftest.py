"""Pytest fixtures for clinical data audit tests."""

import pandas as pd
import pytest


@pytest.fixture
def sample_clinical_df() -> pd.DataFrame:
    """Fixture providing a minimal mock clinical dataset for tests."""
    return pd.DataFrame(
        {
            "patient_id": ["P001", "P002", "P003"],
            "age": [45, 62, 33],
            "gender": ["F", "M", "F"],
            "diagnosis_code": ["E11.9", "I10", "J45.0"],
        }
    )
