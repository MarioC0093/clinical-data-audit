"""Data models for semantic variable type inference."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SemanticType(str, Enum):
    """Semantic statistical roles of clinical variables."""
    IDENTIFIER = "identifier"
    BINARY = "binary"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"


class ConfidenceLevel(str, Enum):
    """Confidence level of the heuristic inference."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class InferenceResult:
    """Result of semantic classification for a single variable."""
    column_name: str
    semantic_type: SemanticType
    confidence: ConfidenceLevel
    score: float
    rationale: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the inference result into a dictionary."""
        return {
            "column_name": self.column_name,
            "semantic_type": self.semantic_type.value,
            "confidence": self.confidence.value,
            "score": self.score,
            "rationale": self.rationale,
            "evidence": self.evidence,
        }
