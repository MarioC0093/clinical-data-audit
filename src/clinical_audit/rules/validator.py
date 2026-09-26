"""Execution engine for configurable quality rules."""

from typing import Dict, List
import pandas as pd

from clinical_audit.rules.models import (
    ConstraintType,
    RuleViolation,
    RulesConfig,
    ValidationReport,
)


def validate_dataframe(df: pd.DataFrame, config: RulesConfig) -> ValidationReport:
    """Evaluates configurable rules against a DataFrame and returns structured violations.

    Important:
        Missing values (NaN, None) are strictly excluded from domain and range
        validations, ensuring that missingness is not conflated with invalid data.

    Args:
        df: Input pandas DataFrame to evaluate.
        config: RulesConfig defining the constraints for each column.

    Returns:
        ValidationReport: Summary of executed checks and list of RuleViolation objects.

    Raises:
        TypeError: If `df` is not a DataFrame or `config` is not a RulesConfig instance.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if not isinstance(config, RulesConfig):
        raise TypeError(f"Expected RulesConfig, got {type(config).__name__}.")

    violations: List[RuleViolation] = []
    violations_by_column: Dict[str, int] = {col: 0 for col in config.columns}

    for col_name, rule_config in config.columns.items():
        if col_name not in df.columns:
            violation = RuleViolation(
                column=col_name,
                row_index=-1,
                invalid_value=None,
                rule_type="missing_column",
                constraint="Column exists in dataset",
                message=f"Configured column '{col_name}' is missing from the dataset.",
            )
            violations.append(violation)
            violations_by_column[col_name] = violations_by_column.get(col_name, 0) + 1
            continue

        series = df[col_name]

        # EXCLUDE missing values so that missingness is NOT treated as invalid
        not_null_mask = series.notna()
        if not not_null_mask.any():
            continue

        non_null_series = series[not_null_mask]

        if rule_config.rule_type == ConstraintType.NUMERIC:
            num_cfg = rule_config.numeric
            if not num_cfg:
                continue

            # Convert to numeric, identifying any non-numeric strings
            numeric_converted = pd.to_numeric(non_null_series, errors="coerce")

            for idx, raw_val in non_null_series.items():
                parsed_val = numeric_converted.loc[idx]

                if pd.isna(parsed_val):
                    # Value was not null, but could not be parsed as a number
                    violation = RuleViolation(
                        column=col_name,
                        row_index=int(idx) if isinstance(idx, (int, float)) else idx,
                        invalid_value=raw_val,
                        rule_type="numeric_type",
                        constraint="Must be a valid numeric value",
                        message=f"Non-numeric value '{raw_val}' in column '{col_name}'.",
                    )
                    violations.append(violation)
                    violations_by_column[col_name] += 1
                    continue

                # Check minimum constraint
                if num_cfg.min_value is not None and parsed_val < num_cfg.min_value:
                    violation = RuleViolation(
                        column=col_name,
                        row_index=int(idx) if isinstance(idx, (int, float)) else idx,
                        invalid_value=raw_val,
                        rule_type="numeric_min",
                        constraint=f"min: {num_cfg.min_value}",
                        message=(
                            f"Value {raw_val} in column '{col_name}' is below "
                            f"allowed minimum of {num_cfg.min_value}."
                        ),
                    )
                    violations.append(violation)
                    violations_by_column[col_name] += 1

                # Check maximum constraint
                if num_cfg.max_value is not None and parsed_val > num_cfg.max_value:
                    violation = RuleViolation(
                        column=col_name,
                        row_index=int(idx) if isinstance(idx, (int, float)) else idx,
                        invalid_value=raw_val,
                        rule_type="numeric_max",
                        constraint=f"max: {num_cfg.max_value}",
                        message=(
                            f"Value {raw_val} in column '{col_name}' exceeds "
                            f"allowed maximum of {num_cfg.max_value}."
                        ),
                    )
                    violations.append(violation)
                    violations_by_column[col_name] += 1

        elif rule_config.rule_type == ConstraintType.CATEGORICAL:
            cat_cfg = rule_config.categorical
            if not cat_cfg:
                continue

            allowed_set = set(cat_cfg.allowed)
            # Support both exact typed match and string representation match
            allowed_strings = {str(item) for item in cat_cfg.allowed}

            for idx, raw_val in non_null_series.items():
                if raw_val not in allowed_set and str(raw_val) not in allowed_strings:
                    violation = RuleViolation(
                        column=col_name,
                        row_index=int(idx) if isinstance(idx, (int, float)) else idx,
                        invalid_value=raw_val,
                        rule_type="categorical_allowed",
                        constraint=f"allowed: {cat_cfg.allowed}",
                        message=(
                            f"Value '{raw_val}' in column '{col_name}' is not in "
                            f"allowed categories {cat_cfg.allowed}."
                        ),
                    )
                    violations.append(violation)
                    violations_by_column[col_name] += 1

    return ValidationReport(
        total_records=len(df),
        total_violations=len(violations),
        violations_by_column=violations_by_column,
        violations=violations,
    )
