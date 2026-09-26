"""Data ingestion and parsing for clinical CSV files."""

from pathlib import Path
from typing import BinaryIO, Union
import pandas as pd


def load_clinical_data(source: Union[str, Path, BinaryIO]) -> pd.DataFrame:
    """Loads a clinical CSV dataset from a file path or file-like buffer.

    Args:
        source: File path (str/Path) or file-like object (e.g. io.BytesIO, UploadedFile).

    Returns:
        pd.DataFrame containing the loaded dataset.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty or cannot be parsed as a CSV.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The provided CSV file is empty.") from exc
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file: {exc}") from exc

    return df
