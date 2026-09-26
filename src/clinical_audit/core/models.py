"""Domain models for clinical data audit."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    """Severity levels for audit findings."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class RuleStatus(str, Enum):
    """Evaluation status of a quality rule."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


@dataclass
class DatasetSummary:
    """High-level structural summary of an audited dataset."""
    row_count: int
    column_count: int
    columns: List[str]
    file_name: Optional[str] = None


@dataclass
class RuleResult:
    """Outcome of a single quality rule evaluation."""
    rule_id: str
    rule_name: str
    category: str
    status: RuleStatus
    severity: Severity
    message: str
    affected_rows_count: int = 0
    affected_columns: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    """Complete report combining dataset summary and quality rule results."""
    dataset_summary: DatasetSummary
    results: List[RuleResult] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.status == RuleStatus.PASSED)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if r.status == RuleStatus.FAILED)

    @property
    def warning_count(self) -> int:
        return sum(1 for r in self.results if r.status == RuleStatus.WARNING)


@dataclass
class ColumnQualitySummary:
    """Summary of data quality and structure for an individual column."""
    name: str
    dtype: str
    missing_count: int
    missing_percentage: float
    unique_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the column summary to a dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "missing_count": self.missing_count,
            "missing_percentage": self.missing_percentage,
            "unique_count": self.unique_count,
        }


@dataclass
class QualityReport:
    """Structured quality report covering dataset-level and column-level metrics."""
    row_count: int
    column_count: int
    exact_duplicates_count: int
    columns: Dict[str, ColumnQualitySummary] = field(default_factory=dict)
    file_name: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the complete quality report to a dictionary."""
        return {
            "file_name": self.file_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "exact_duplicates_count": self.exact_duplicates_count,
            "columns": {name: col.to_dict() for name, col in self.columns.items()},
            "created_at": self.created_at.isoformat(),
        }
