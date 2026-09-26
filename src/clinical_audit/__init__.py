"""Clinical Data Audit package.

Core package for analyzing, validating, and auditing clinical datasets.
"""

__version__ = "0.1.0"

from clinical_audit.core.models import ColumnQualitySummary, QualityReport
from clinical_audit.inference.classifier import (
    infer_column_type,
    infer_dataset_types,
)
from clinical_audit.inference.models import (
    ConfidenceLevel,
    InferenceResult,
    SemanticType,
)
from clinical_audit.quality.profiler import generate_quality_report
from clinical_audit.rules.config import load_rules_from_yaml
from clinical_audit.rules.models import RuleViolation, RulesConfig, ValidationReport
from clinical_audit.rules.validator import validate_dataframe

__all__ = [
    "ColumnQualitySummary",
    "ConfidenceLevel",
    "InferenceResult",
    "QualityReport",
    "RuleViolation",
    "RulesConfig",
    "SemanticType",
    "ValidationReport",
    "generate_quality_report",
    "infer_column_type",
    "infer_dataset_types",
    "load_rules_from_yaml",
    "validate_dataframe",
]
