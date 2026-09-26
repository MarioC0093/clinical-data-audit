"""Models for YAML-configurable quality rules and validation violations."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ConstraintType(str, Enum):
    """Supported rule constraint types."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


@dataclass
class NumericRuleConfig:
    """Constraints for numeric columns."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class CategoricalRuleConfig:
    """Constraints for categorical columns."""
    allowed: List[Any] = field(default_factory=list)


@dataclass
class ColumnRuleConfig:
    """Configuration of rules applicable to a specific column."""
    column_name: str
    rule_type: ConstraintType
    numeric: Optional[NumericRuleConfig] = None
    categorical: Optional[CategoricalRuleConfig] = None


@dataclass
class RulesConfig:
    """Collection of column-level validation rules."""
    columns: Dict[str, ColumnRuleConfig] = field(default_factory=dict)


@dataclass
class RuleViolation:
    """Structured record of a single data validation failure."""
    column: str
    row_index: int
    invalid_value: Any
    rule_type: str
    constraint: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the violation into a plain dictionary."""
        return {
            "column": self.column,
            "row_index": self.row_index,
            "invalid_value": self.invalid_value,
            "rule_type": self.rule_type,
            "constraint": self.constraint,
            "message": self.message,
        }


@dataclass
class ValidationReport:
    """Comprehensive outcome of executing configurable rules against a dataset."""
    total_records: int
    total_violations: int
    violations_by_column: Dict[str, int]
    violations: List[RuleViolation] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True if no rule violations were detected."""
        return self.total_violations == 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the validation report to a dictionary."""
        return {
            "total_records": self.total_records,
            "total_violations": self.total_violations,
            "is_valid": self.is_valid,
            "violations_by_column": self.violations_by_column,
            "violations": [v.to_dict() for v in self.violations],
        }
