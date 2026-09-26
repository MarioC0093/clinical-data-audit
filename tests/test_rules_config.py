"""Unit tests for YAML rule configuration parsing and schema validation."""

from pathlib import Path
import pytest

from clinical_audit.rules.config import load_rules_from_yaml
from clinical_audit.rules.models import ConstraintType, RulesConfig


def test_load_rules_from_valid_yaml_file() -> None:
    """Tests loading the sample clinical rules YAML file."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "clinical_rules.yaml"
    assert config_path.exists(), f"clinical_rules.yaml not found at {config_path}"

    config = load_rules_from_yaml(config_path)
    assert isinstance(config, RulesConfig)
    assert len(config.columns) == 5

    # Check numeric rule structure
    age_cfg = config.columns["age"]
    assert age_cfg.rule_type == ConstraintType.NUMERIC
    assert age_cfg.numeric is not None
    assert age_cfg.numeric.min_value == 0.0
    assert age_cfg.numeric.max_value == 120.0

    # Check categorical rule structure
    smoker_cfg = config.columns["smoker"]
    assert smoker_cfg.rule_type == ConstraintType.CATEGORICAL
    assert smoker_cfg.categorical is not None
    assert smoker_cfg.categorical.allowed == ["Yes", "No"]


def test_load_rules_from_yaml_string() -> None:
    """Tests parsing YAML directly from a string."""
    yaml_text = """
    heart_rate:
      type: numeric
      min: 30
      max: 220
    triage_level:
      type: categorical
      allowed:
        - 1
        - 2
        - 3
    """
    config = load_rules_from_yaml(yaml_text)
    assert "heart_rate" in config.columns
    assert config.columns["heart_rate"].numeric.min_value == 30.0
    assert config.columns["triage_level"].categorical.allowed == [1, 2, 3]


def test_load_rules_file_not_found() -> None:
    """Tests that FileNotFoundError is raised for non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_rules_from_yaml(Path("non_existent_rules.yaml"))


def test_load_rules_invalid_yaml_syntax() -> None:
    """Tests that ValueError is raised for invalid YAML syntax."""
    invalid_yaml = "age: [unbalanced"
    with pytest.raises(ValueError, match="Failed to parse YAML"):
        load_rules_from_yaml(invalid_yaml)


def test_load_rules_missing_type_field() -> None:
    """Tests that ValueError is raised when 'type' is missing."""
    yaml_text = """
    age:
      min: 0
    """
    with pytest.raises(ValueError, match="missing required 'type' field"):
        load_rules_from_yaml(yaml_text)


def test_load_rules_unsupported_type() -> None:
    """Tests that ValueError is raised for unsupported constraint type."""
    yaml_text = """
    age:
      type: boolean
    """
    with pytest.raises(ValueError, match="Invalid type 'boolean'"):
        load_rules_from_yaml(yaml_text)


def test_load_rules_numeric_min_greater_than_max() -> None:
    """Tests range sanity check where min > max."""
    yaml_text = """
    age:
      type: numeric
      min: 100
      max: 50
    """
    with pytest.raises(ValueError, match="min .* cannot be greater than max"):
        load_rules_from_yaml(yaml_text)


def test_load_rules_categorical_allowed_not_a_list() -> None:
    """Tests that 'allowed' must be a list."""
    yaml_text = """
    sex:
      type: categorical
      allowed: "F, M"
    """
    with pytest.raises(ValueError, match="'allowed' as a list"):
        load_rules_from_yaml(yaml_text)


def test_load_rules_empty_content() -> None:
    """Tests empty YAML string returns empty RulesConfig."""
    config = load_rules_from_yaml("")
    assert isinstance(config, RulesConfig)
    assert len(config.columns) == 0
