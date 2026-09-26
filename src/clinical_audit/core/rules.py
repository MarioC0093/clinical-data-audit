"""Base interfaces and contracts for quality rules."""

from abc import ABC, abstractmethod
import pandas as pd

from clinical_audit.core.models import RuleResult, Severity


class BaseRule(ABC):
    """Abstract base class for all clinical data quality rules."""

    rule_id: str = "BASE_RULE"
    rule_name: str = "Base Quality Rule"
    category: str = "General"
    severity: Severity = Severity.WARNING
    description: str = "Base quality rule template"

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> RuleResult:
        """Evaluates the rule against the provided clinical dataframe.

        Args:
            df: Clinical pandas DataFrame to analyze.

        Returns:
            RuleResult detailing whether the rule passed or failed.
        """
        pass
