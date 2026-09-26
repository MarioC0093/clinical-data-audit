"""Clinical quality checks collection.

This module houses quality checks grouped by categories (e.g. completeness,
validity, clinical range, temporal coherence).
"""

from clinical_audit.checks.base import BaseRule

__all__ = ["BaseRule"]
