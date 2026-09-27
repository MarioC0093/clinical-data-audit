"""Tests that verify all CSV files in the data/ directory can be loaded.

Each CSV file is discovered automatically so new samples are tested without
any manual changes to this file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Make sure src/ is importable when running pytest from the repo root
import sys
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clinical_audit.ingestion.loader import load_clinical_data

# ── Discover sample CSV files ─────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CSV_FILES = sorted(DATA_DIR.glob("*.csv"))


# ── Fixtures / parametrization ────────────────────────────────────────────────

def _csv_id(path: Path) -> str:
    """Human-readable test ID (just the filename)."""
    return path.name


@pytest.mark.parametrize("csv_path", CSV_FILES, ids=_csv_id)
class TestCsvSamples:
    """Parametrised suite — one test instance per CSV file in data/."""

    def test_file_exists(self, csv_path: Path) -> None:
        """The CSV file must physically exist on disk."""
        assert csv_path.exists(), f"Expected file not found: {csv_path}"

    def test_loads_without_error(self, csv_path: Path) -> None:
        """load_clinical_data() must return a DataFrame without raising."""
        df = load_clinical_data(csv_path)
        assert isinstance(df, pd.DataFrame)

    def test_has_rows(self, csv_path: Path) -> None:
        """Loaded DataFrame must contain at least one data row."""
        df = load_clinical_data(csv_path)
        assert df.shape[0] > 0, f"{csv_path.name}: no rows loaded"

    def test_has_columns(self, csv_path: Path) -> None:
        """Loaded DataFrame must have more than one column (separator detected)."""
        df = load_clinical_data(csv_path)
        assert df.shape[1] > 1, (
            f"{csv_path.name}: only 1 column loaded — separator may not have been detected"
        )

    def test_column_names_are_strings(self, csv_path: Path) -> None:
        """All column names must be strings."""
        df = load_clinical_data(csv_path)
        for col in df.columns:
            assert isinstance(col, str), f"Non-string column name: {col!r}"


# ── Extra tests for the known sample file ─────────────────────────────────────

class TestClinicalSampleCsv:
    """Deeper assertions for the bundled clinical_sample.csv fixture."""

    SAMPLE_PATH = DATA_DIR / "clinical_sample.csv"

    @pytest.fixture(autouse=True)
    def skip_if_missing(self) -> None:
        if not self.SAMPLE_PATH.exists():
            pytest.skip("clinical_sample.csv not found in data/")

    def test_expected_columns_present(self) -> None:
        df = load_clinical_data(self.SAMPLE_PATH)
        expected = {"patient_id", "age", "sex", "bmi"}
        missing = expected - set(df.columns)
        assert not missing, f"Expected columns missing: {missing}"

    def test_age_column_is_numeric(self) -> None:
        df = load_clinical_data(self.SAMPLE_PATH)
        age_numeric = pd.to_numeric(df["age"], errors="coerce")
        assert age_numeric.notna().mean() > 0.9, "age column is not predominantly numeric"


class TestHHENCsv:
    """Assertions for the HHEN anxiety/depression dataset (semicolon + latin-1)."""

    HHEN_FILES = list(DATA_DIR.glob("HHEN*.csv"))

    def test_hhen_file_present(self) -> None:
        assert self.HHEN_FILES, "No HHEN*.csv file found in data/"

    def test_hhen_loads_with_multiple_columns(self) -> None:
        for p in self.HHEN_FILES:
            df = load_clinical_data(p)
            assert df.shape[1] > 1, f"{p.name}: separator not detected, only 1 column"
            assert df.shape[0] > 0, f"{p.name}: no rows loaded"
