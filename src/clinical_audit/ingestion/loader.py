"""Data ingestion and parsing for clinical CSV files.

Supports automatic encoding and separator detection so that files
encoded in UTF-8, latin-1 (ISO-8859-1), cp1252, etc. as well as
files using comma, semicolon or tab separators are loaded correctly.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, List, Union

import pandas as pd


# Encodings tried in order (most common first)
_ENCODINGS: List[str] = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-15"]

# Separators tried in order
_SEPARATORS: List[str] = [",", ";", "\t", "|"]


def load_clinical_data(source: Union[str, Path, BinaryIO]) -> pd.DataFrame:
    """Loads a clinical CSV dataset from a file path or file-like buffer.

    Automatically tries multiple character encodings and column separators
    so that files with different regional conventions are handled robustly.

    Args:
        source: File path (str/Path) or file-like object (e.g. io.BytesIO,
                Streamlit UploadedFile).

    Returns:
        pd.DataFrame containing the loaded dataset.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the file is empty or cannot be parsed as a CSV with
                    any of the supported encoding/separator combinations.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    last_error: Exception = ValueError("Unknown error during CSV parsing.")

    for encoding in _ENCODINGS:
        for sep in _SEPARATORS:
            try:
                # Reset stream position for file-like objects on each attempt
                if hasattr(source, "seek"):
                    source.seek(0)

                df = pd.read_csv(source, encoding=encoding, sep=sep)

                # Reject single-column frames when other separators may work
                # (avoid accepting a badly split file)
                if df.shape[1] == 1 and sep != _SEPARATORS[-1]:
                    continue

                if df.empty:
                    raise ValueError("The provided CSV file is empty.")

                return df

            except (UnicodeDecodeError, pd.errors.ParserError):
                # Try next encoding/separator combination
                continue
            except pd.errors.EmptyDataError as exc:
                raise ValueError("The provided CSV file is empty.") from exc
            except Exception as exc:
                last_error = exc
                # Continue trying other combinations
                continue

    raise ValueError(
        f"Failed to parse CSV file with any supported encoding "
        f"({', '.join(_ENCODINGS)}) or separator. Last error: {last_error}"
    )
