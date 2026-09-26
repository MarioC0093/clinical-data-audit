"""Formatting and presentation layer for audit reports."""

from typing import Any, Dict
from clinical_audit.core.models import AuditReport


def report_to_dict(report: AuditReport) -> Dict[str, Any]:
    """Converts an AuditReport into a serialized dictionary format."""
    return {
        "dataset_summary": {
            "file_name": report.dataset_summary.file_name,
            "row_count": report.dataset_summary.row_count,
            "column_count": report.dataset_summary.column_count,
            "columns": report.dataset_summary.columns,
        },
        "stats": {
            "total_rules": len(report.results),
            "passed": report.passed_count,
            "failed": report.failed_count,
            "warnings": report.warning_count,
        },
        "results": [
            {
                "rule_id": r.rule_id,
                "rule_name": r.rule_name,
                "category": r.category,
                "status": r.status.value,
                "severity": r.severity.value,
                "message": r.message,
                "affected_rows_count": r.affected_rows_count,
                "affected_columns": r.affected_columns,
                "details": r.details,
            }
            for r in report.results
        ],
        "created_at": report.created_at.isoformat(),
    }


def report_to_markdown(report: AuditReport) -> str:
    """Renders a simple Markdown representation of an AuditReport."""
    lines = [
        "# Clinical Data Audit Report",
        "",
        f"- **File**: {report.dataset_summary.file_name or 'In-memory buffer'}",
        f"- **Rows**: {report.dataset_summary.row_count:,}",
        f"- **Columns**: {report.dataset_summary.column_count}",
        f"- **Timestamp (UTC)**: {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        f"- Total Rules Evaluated: {len(report.results)}",
        f"- Passed: {report.passed_count}",
        f"- Failed: {report.failed_count}",
        f"- Warnings: {report.warning_count}",
        "",
        "## Detailed Findings",
    ]

    if not report.results:
        lines.append("*No quality rules were executed.*")
    else:
        for r in report.results:
            lines.append(
                f"- **[{r.status.value}]** `{r.rule_id}`: {r.rule_name} - {r.message}"
            )

    return "\n".join(lines)
