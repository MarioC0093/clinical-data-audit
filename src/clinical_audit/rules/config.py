"""Parser and loader for YAML-based validation rule configurations."""

from pathlib import Path
from typing import Any, Dict, Union
import yaml

from clinical_audit.rules.models import (
    CategoricalRuleConfig,
    ColumnRuleConfig,
    ConstraintType,
    NumericRuleConfig,
    RulesConfig,
)


class ClinicalRulesYamlLoader(yaml.SafeLoader):
    """Custom YAML loader that treats unquoted Yes/No as strings (avoiding boolean conversion)."""
    pass


# Disassociate y, Y, n, N, o, O from boolean resolver in custom loader
ClinicalRulesYamlLoader.yaml_implicit_resolvers = {
    k: [
        (tag, reg)
        for tag, reg in v
        if not (tag == "tag:yaml.org,2002:bool" and k in ["y", "Y", "n", "N", "o", "O"])
    ]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def load_rules_from_yaml(source: Union[str, Path]) -> RulesConfig:
    """Parses and validates a YAML rule specification into a RulesConfig object.

    Args:
        source: File path (str or Path) or a raw YAML string.

    Returns:
        RulesConfig: Validated strongly-typed configuration of column rules.

    Raises:
        FileNotFoundError: If the source is a file path that does not exist.
        ValueError: If YAML syntax is invalid or structure doesn't meet expected schema.
    """
    raw_content: str
    if isinstance(source, Path):
        if not source.exists():
            raise FileNotFoundError(f"Configuration file not found: {source}")
        raw_content = source.read_text(encoding="utf-8")
    elif isinstance(source, str):
        path = Path(source)
        if "\n" not in source and path.exists() and path.is_file():
            raw_content = path.read_text(encoding="utf-8")
        else:
            raw_content = source
    else:
        raise TypeError(f"Expected str or Path, got {type(source).__name__}")

    try:
        parsed_data = yaml.load(raw_content, Loader=ClinicalRulesYamlLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML configuration: {exc}") from exc

    if parsed_data is None:
        return RulesConfig(columns={})

    if not isinstance(parsed_data, dict):
        raise ValueError(
            "Root of YAML rules configuration must be a mapping of column names to rule definitions."
        )

    columns: Dict[str, ColumnRuleConfig] = {}

    for col_name, rule_def in parsed_data.items():
        if not isinstance(rule_def, dict):
            raise ValueError(
                f"Rule definition for column '{col_name}' must be a mapping/dictionary."
            )

        raw_type = rule_def.get("type")
        if not raw_type:
            raise ValueError(f"Column '{col_name}' missing required 'type' field.")

        try:
            rule_type = ConstraintType(str(raw_type).lower())
        except ValueError:
            raise ValueError(
                f"Invalid type '{raw_type}' for column '{col_name}'. "
                f"Supported types are: {[t.value for t in ConstraintType]}."
            )

        if rule_type == ConstraintType.NUMERIC:
            min_val = rule_def.get("min")
            max_val = rule_def.get("max")

            if min_val is None and max_val is None:
                raise ValueError(
                    f"Numeric rule for column '{col_name}' must specify at least 'min' or 'max'."
                )

            min_float = float(min_val) if min_val is not None else None
            max_float = float(max_val) if max_val is not None else None

            if min_float is not None and max_float is not None and min_float > max_float:
                raise ValueError(
                    f"Column '{col_name}' has invalid range: min ({min_float}) cannot be greater than max ({max_float})."
                )

            columns[col_name] = ColumnRuleConfig(
                column_name=col_name,
                rule_type=rule_type,
                numeric=NumericRuleConfig(min_value=min_float, max_value=max_float),
            )

        elif rule_type == ConstraintType.CATEGORICAL:
            allowed = rule_def.get("allowed")
            if not isinstance(allowed, list):
                raise ValueError(
                    f"Categorical rule for column '{col_name}' must specify 'allowed' as a list of acceptable values."
                )

            columns[col_name] = ColumnRuleConfig(
                column_name=col_name,
                rule_type=rule_type,
                categorical=CategoricalRuleConfig(allowed=allowed),
            )

    return RulesConfig(columns=columns)
