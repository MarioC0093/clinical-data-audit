"""Heuristic semantic type classifier for clinical dataset variables."""

import re
from typing import Any, Dict, Optional
import pandas as pd

from clinical_audit.inference.models import (
    ConfidenceLevel,
    InferenceResult,
    SemanticType,
)

# Regex patterns for variable name hints
ID_NAME_PATTERN = re.compile(
    r"(^|[_.-])(id|patient|nhc|uuid|identifier|record|cod|codigo|subject|mrn)([_.-]|$)",
    re.IGNORECASE,
)
BINARY_NAME_PATTERN = re.compile(
    r"(^|[_.-])(is_|has_|flag|smoker|fumador|sex|genero|gender|fallecido|deceased|active|activo)([_.-]|$)",
    re.IGNORECASE,
)
BIOMETRIC_CONTINUOUS_PATTERN = re.compile(
    r"(^|[_.-])(bp|presion|pressure|systolic|diastolic|bmi|imc|peso|weight|talla|height|temp|temperatura|glucose|glucosa|cholesterol|colesterol|ratio|tasa|rate)([_.-]|$)",
    re.IGNORECASE,
)
DISCRETE_COUNT_PATTERN = re.compile(
    r"(^|[_.-])(num|count|n_|visitas|admissions|episodios|partos|hijos|dias|days|veces|times|ingresos)([_.-]|$)",
    re.IGNORECASE,
)
AGE_NAME_PATTERN = re.compile(
    r"(^|[_.-])(age|edad)([_.-]|$)",
    re.IGNORECASE,
)

# Standard binary token sets
KNOWN_BINARY_SETS = {
    frozenset({"0", "1"}),
    frozenset({"0.0", "1.0"}),
    frozenset({"true", "false"}),
    frozenset({"yes", "no"}),
    frozenset({"si", "no"}),
    frozenset({"s", "n"}),
    frozenset({"m", "f"}),
    frozenset({"male", "female"}),
    frozenset({"hombre", "mujer"}),
    frozenset({"pos", "neg"}),
    frozenset({"positive", "negative"}),
}


def _is_alphanumeric_id(series: pd.Series) -> bool:
    """Checks if values look like alphanumeric codes (e.g., P001, PAT-123)."""
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    pattern = re.compile(r"^[A-Za-z]+[-_]?\d+$")
    matches = sum(1 for val in sample if pattern.match(val.strip()))
    return (matches / len(sample)) >= 0.8


def infer_column_type(
    series: pd.Series,
    column_name: Optional[str] = None,
) -> InferenceResult:
    """Infers the semantic statistical type of a clinical variable using heuristics.

    Evaluates variable name, cardinality, uniqueness ratio, floating decimals,
    value distributions, and domain semantics without relying solely on pandas dtype.

    Args:
        series: Pandas Series containing the variable values.
        column_name: Optional override for the column name.

    Returns:
        InferenceResult containing the semantic type, confidence, score, and rationale.
    """
    col_name = str(column_name or series.name or "unnamed_column")
    non_null = series.dropna()
    n_total = len(series)
    n_valid = len(non_null)

    # Base evidence dictionary
    evidence: Dict[str, Any] = {
        "total_rows": n_total,
        "valid_rows": n_valid,
        "null_rows": n_total - n_valid,
    }

    # Edge Case: Completely empty column
    if n_valid == 0:
        return InferenceResult(
            column_name=col_name,
            semantic_type=SemanticType.CATEGORICAL_NOMINAL,
            confidence=ConfidenceLevel.LOW,
            score=0.20,
            rationale="Columna vacía (todos los registros son nulos); no hay evidencia concluyente.",
            evidence=evidence,
        )

    n_unique = int(non_null.nunique())
    unique_ratio = n_unique / n_valid if n_valid > 0 else 0.0
    evidence["unique_count"] = n_unique
    evidence["unique_ratio"] = round(unique_ratio, 4)

    # Check numeric conversion
    numeric_converted = pd.to_numeric(non_null, errors="coerce")
    is_numeric = bool(numeric_converted.notna().all())
    evidence["is_numeric"] = is_numeric

    # 1. EVALUATION: IDENTIFIERS
    name_is_id = bool(ID_NAME_PATTERN.search(col_name))
    value_looks_like_id = _is_alphanumeric_id(non_null)
    evidence["name_has_id_pattern"] = name_is_id
    evidence["values_look_like_id"] = value_looks_like_id

    # Strong ID: 100% unique with ID name or alphanumeric code pattern
    if unique_ratio == 1.0 and (name_is_id or value_looks_like_id):
        return InferenceResult(
            column_name=col_name,
            semantic_type=SemanticType.IDENTIFIER,
            confidence=ConfidenceLevel.HIGH,
            score=0.98,
            rationale=f"100% de unicidad y patrón de identificador detectado ('{col_name}').",
            evidence=evidence,
        )

    # Longitudinal / Multiepisodic ID (same patient recorded multiple times)
    if name_is_id and (value_looks_like_id or unique_ratio >= 0.50) and n_unique > 2:
        return InferenceResult(
            column_name=col_name,
            semantic_type=SemanticType.IDENTIFIER,
            confidence=ConfidenceLevel.MEDIUM,
            score=0.80,
            rationale=(
                f"Nombre y formato de identificador ('{col_name}') con repeticiones "
                f"({n_unique} únicos en {n_valid} filas, ratio {unique_ratio:.2f}); "
                "compatible con identificador longitudinal o multiepisodio."
            ),
            evidence=evidence,
        )

    # Sequential integer ID starting at 0 or 1 with ID name
    if (
        is_numeric
        and name_is_id
        and unique_ratio == 1.0
        and (numeric_converted % 1 == 0).all()
    ):
        return InferenceResult(
            column_name=col_name,
            semantic_type=SemanticType.IDENTIFIER,
            confidence=ConfidenceLevel.HIGH,
            score=0.95,
            rationale="Secuencia numérica entera incremental con nombre característico de identificador.",
            evidence=evidence,
        )

    # 2. EVALUATION: BINARY VARIABLES
    str_vals_set = {str(x).strip().lower() for x in non_null.unique()}
    evidence["unique_string_values"] = sorted(list(str_vals_set))

    if n_unique == 2:
        # Standard recognized binary domain
        if any(str_vals_set.issubset(b_set) for b_set in KNOWN_BINARY_SETS):
            return InferenceResult(
                column_name=col_name,
                semantic_type=SemanticType.BINARY,
                confidence=ConfidenceLevel.HIGH,
                score=0.98,
                rationale=f"Exactamente 2 categorías pertenecientes a un dominio dicotómico estándar: {sorted(list(str_vals_set))}.",
                evidence=evidence,
            )
        # Any non-numeric 2 categories
        if not is_numeric:
            return InferenceResult(
                column_name=col_name,
                semantic_type=SemanticType.BINARY,
                confidence=ConfidenceLevel.HIGH,
                score=0.92,
                rationale=f"Variable dicotómica no numérica con exactamente 2 categorías observadas: {sorted(list(str_vals_set))}.",
                evidence=evidence,
            )
        # Numeric 0/1
        if str_vals_set in ({"0", "1"}, {"0.0", "1.0"}):
            return InferenceResult(
                column_name=col_name,
                semantic_type=SemanticType.BINARY,
                confidence=ConfidenceLevel.HIGH,
                score=0.96,
                rationale="Variable binaria numérica codificada como 0 y 1.",
                evidence=evidence,
            )
        # Numeric with 2 values, but arbitrary numbers (e.g. 10 and 20) -> Ambiguous
        return InferenceResult(
            column_name=col_name,
            semantic_type=SemanticType.BINARY,
            confidence=ConfidenceLevel.MEDIUM,
            score=0.72,
            rationale=f"Variable numérica con solo 2 valores únicos observados: {sorted(list(str_vals_set))}.",
            evidence=evidence,
        )

    # Ambiguous case: Binary with third corrupted/impure category (e.g. Yes, No, Maybe)
    if n_unique == 3:
        binary_matches = sum(
            1 for b_set in KNOWN_BINARY_SETS if len(str_vals_set.intersection(b_set)) == 2
        )
        if binary_matches > 0:
            return InferenceResult(
                column_name=col_name,
                semantic_type=SemanticType.BINARY,
                confidence=ConfidenceLevel.LOW,
                score=0.55,
                rationale=(
                    f"Columna con 3 categorías que contiene un par binario reconocible "
                    f"junto a un tercer valor probablemente anómalo o indeterminado: {sorted(list(str_vals_set))}."
                ),
                evidence=evidence,
            )

    # 3. EVALUATION: NUMERIC CONTINUOUS VS NUMERIC DISCRETE
    if is_numeric:
        has_decimals = bool(((numeric_converted % 1) != 0).any())
        all_integers = not has_decimals
        min_val = float(numeric_converted.min())
        max_val = float(numeric_converted.max())
        evidence["has_decimals"] = has_decimals
        evidence["all_integers"] = all_integers
        evidence["min_value"] = min_val
        evidence["max_value"] = max_val

        # Clear Continuous: presence of floating decimals
        if has_decimals:
            return InferenceResult(
                column_name=col_name,
                semantic_type=SemanticType.NUMERIC_CONTINUOUS,
                confidence=ConfidenceLevel.HIGH,
                score=0.96,
                rationale="Valores numéricos con parte decimal fraccionaria; magnitud continua evidente.",
                evidence=evidence,
            )

        # Ambiguous Domain Case: Age (recorded in integer years, but conceptually continuous time)
        if AGE_NAME_PATTERN.search(col_name):
            return InferenceResult(
                column_name=col_name,
                semantic_type=SemanticType.NUMERIC_CONTINUOUS,
                confidence=ConfidenceLevel.MEDIUM,
                score=0.75,
                rationale=(
                    "Edad (age): magnitud biológica continua habitualmente registrada "
                    "en años enteros discretizados."
                ),
                evidence=evidence,
            )

        # Ambiguous Domain Case: Continuous biometrics measured in whole integers (e.g., systolic_bp)
        if BIOMETRIC_CONTINUOUS_PATTERN.search(col_name):
            return InferenceResult(
                column_name=col_name,
                semantic_type=SemanticType.NUMERIC_CONTINUOUS,
                confidence=ConfidenceLevel.MEDIUM,
                score=0.82,
                rationale=(
                    f"Parámetro clínico biometríco continuo ('{col_name}') "
                    f"registrado en unidades enteras (rango: [{min_val:.0f}, {max_val:.0f}])."
                ),
                evidence=evidence,
            )

        # Clear Discrete: Name indicates counting / frequency
        if DISCRETE_COUNT_PATTERN.search(col_name) and all_integers and min_val >= 0:
            return InferenceResult(
                column_name=col_name,
                semantic_type=SemanticType.NUMERIC_DISCRETE,
                confidence=ConfidenceLevel.HIGH,
                score=0.94,
                rationale=f"Nombre de conteo/frecuencia ('{col_name}') con valores enteros no negativos.",
                evidence=evidence,
            )

        # Moderate cardinality integers with small non-negative values -> Discrete counts
        if all_integers and min_val >= 0 and max_val <= 30 and n_unique <= 15:
            return InferenceResult(
                column_name=col_name,
                semantic_type=SemanticType.NUMERIC_DISCRETE,
                confidence=ConfidenceLevel.MEDIUM,
                score=0.75,
                rationale=(
                    f"Enteros no negativos en rango acotado [{int(min_val)}, {int(max_val)}] "
                    f"con {n_unique} valores únicos; distribución típica de recuentos discretos."
                ),
                evidence=evidence,
            )

        # High cardinality integer values (e.g. cholesterol, platelets, heart rate without specific names)
        if all_integers and n_unique > 15:
            return InferenceResult(
                column_name=col_name,
                semantic_type=SemanticType.NUMERIC_CONTINUOUS,
                confidence=ConfidenceLevel.MEDIUM,
                score=0.70,
                rationale=(
                    f"Variable numérica entera con cardinalidad alta ({n_unique} únicos, "
                    f"rango [{int(min_val)}, {int(max_val)}]); operativamente continua."
                ),
                evidence=evidence,
            )

        # Low cardinality integers without count hints (e.g. values 1, 2, 3) -> Ambiguous Discrete / Nominal
        return InferenceResult(
            column_name=col_name,
            semantic_type=SemanticType.NUMERIC_DISCRETE,
            confidence=ConfidenceLevel.LOW,
            score=0.58,
            rationale=(
                f"Enteros con baja cardinalidad ({n_unique} únicos en [{int(min_val)}, {int(max_val)}]); "
                "podría representar recuentos discretos o categorías ordinales/codificadas."
            ),
            evidence=evidence,
        )

    # 4. EVALUATION: CATEGORICAL NOMINAL
    # Text / string data with more than 2 distinct values
    return InferenceResult(
        column_name=col_name,
        semantic_type=SemanticType.CATEGORICAL_NOMINAL,
        confidence=ConfidenceLevel.HIGH,
        score=0.92,
        rationale=f"Variable cualitativa de texto con {n_unique} categorías distintas.",
        evidence=evidence,
    )


def infer_dataset_types(df: pd.DataFrame) -> Dict[str, InferenceResult]:
    """Infers semantic types for all columns in a clinical DataFrame.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Dict mapping column name to its InferenceResult.

    Raises:
        TypeError: If input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

    return {col: infer_column_type(df[col], column_name=str(col)) for col in df.columns}
