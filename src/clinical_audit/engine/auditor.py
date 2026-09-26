"""Orchestration engine for clinical data audits."""

from typing import List, Optional
import pandas as pd

from clinical_audit.core.models import AuditReport, DatasetSummary, RuleResult
from clinical_audit.core.rules import BaseRule


class ClinicalAuditor:
    """Coordinates and executes data quality rules against clinical datasets."""

    def __init__(self, rules: Optional[List[BaseRule]] = None) -> None:
        """Initializes the auditor with a set of rules.

        Args:
            rules: Optional list of BaseRule instances.
        """
        self._rules: List[BaseRule] = list(rules) if rules else []

    @property
    def rules(self) -> List[BaseRule]:
        """Returns the registered rules."""
        return list(self._rules)

    def register_rule(self, rule: BaseRule) -> None:
        """Registers an additional quality rule into the auditor.

        Args:
            rule: Instance of BaseRule.
        """
        self._rules.append(rule)

    def audit(self, df: pd.DataFrame, file_name: Optional[str] = None) -> AuditReport:
        """Runs all registered quality rules on the provided dataframe.

        Args:
            df: Clinical pandas DataFrame.
            file_name: Optional name of the audited file.

        Returns:
            AuditReport containing structural summary and rule outcomes.
        """
        summary = DatasetSummary(
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns.astype(str)),
            file_name=file_name,
        )

        results: List[RuleResult] = []
        for rule in self._rules:
            result = rule.evaluate(df)
            results.append(result)

        return AuditReport(
            dataset_summary=summary,
            results=results,
        )
