"""Streamlit UI entrypoint for Clinical Data Audit.

Presentation layer purely consuming core clinical_audit modules without
containing validation or statistical logic.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Ensure src/ is on Python search path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import streamlit as st
import yaml

from clinical_audit.inference.classifier import infer_dataset_types
from clinical_audit.inference.models import SemanticType
from clinical_audit.ingestion.loader import load_clinical_data
from clinical_audit.quality.profiler import generate_quality_report
from clinical_audit.rules.config import load_rules_from_yaml
from clinical_audit.rules.models import (
    CategoricalRuleConfig,
    ColumnRuleConfig,
    ConstraintType,
    NumericRuleConfig,
    RulesConfig,
    ValidationReport,
)
from clinical_audit.rules.validator import validate_dataframe


_APP_DIR = Path(__file__).resolve().parent
DEFAULT_RULES_PATH = _APP_DIR / "config" / "clinical_rules.yaml"
DEFAULT_SAMPLE_DATA_PATH = _APP_DIR / "data" / "clinical_sample.csv"

# Semantic types that map to numeric constraints
_NUMERIC_SEMANTIC_TYPES = {
    SemanticType.NUMERIC_CONTINUOUS,
    SemanticType.NUMERIC_DISCRETE,
}

# Semantic types that map to categorical constraints
_CATEGORICAL_SEMANTIC_TYPES = {
    SemanticType.BINARY,
    SemanticType.CATEGORICAL_NOMINAL,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _rules_config_to_yaml_str(rules_config: RulesConfig) -> str:
    """Serialises a RulesConfig object to a human-readable YAML string."""
    data: Dict = {}
    for col_name, col_rule in rules_config.columns.items():
        if col_rule.rule_type == ConstraintType.NUMERIC and col_rule.numeric:
            entry: Dict = {"type": "numeric"}
            if col_rule.numeric.min_value is not None:
                entry["min"] = col_rule.numeric.min_value
            if col_rule.numeric.max_value is not None:
                entry["max"] = col_rule.numeric.max_value
            data[col_name] = entry
        elif col_rule.rule_type == ConstraintType.CATEGORICAL and col_rule.categorical:
            data[col_name] = {
                "type": "categorical",
                "allowed": list(col_rule.categorical.allowed),
            }
    return yaml.dump(data, allow_unicode=True, sort_keys=True, default_flow_style=False)


def _infer_constraint_type(semantic_type: SemanticType) -> Optional[ConstraintType]:
    """Maps a SemanticType to the most appropriate ConstraintType, or None if ambiguous."""
    if semantic_type in _NUMERIC_SEMANTIC_TYPES:
        return ConstraintType.NUMERIC
    if semantic_type in _CATEGORICAL_SEMANTIC_TYPES:
        return ConstraintType.CATEGORICAL
    return None  # IDENTIFIER → not constrainable


def _render_violations_table(report: ValidationReport) -> None:
    """Renders the violations summary metrics and table."""
    v_col1, v_col2, v_col3 = st.columns(3)
    v_col1.metric("Total Registros Evaluados", report.total_records)
    v_col2.metric(
        "Infracciones Detectadas",
        report.total_violations,
        delta=f"-{report.total_violations}" if report.total_violations > 0 else "0",
        delta_color="inverse",
    )
    v_col3.metric(
        "Estado de Conformidad",
        "APROBADO ✅" if report.is_valid else "NO CONFORME ❌",
    )

    if report.is_valid:
        st.success("🎉 Todas las reglas configuradas se han cumplido sin ninguna infracción.")
    else:
        st.error(
            f"⚠️ Se detectaron {report.total_violations} infracciones en las reglas clínicas."
        )
        violations_data = [
            {
                "Fila (Índice)": v.row_index,
                "Variable": v.column,
                "Valor Observado": str(v.invalid_value),
                "Tipo de Regla": v.rule_type,
                "Restricción Esperada": v.constraint,
                "Detalle del Fallo": v.message,
            }
            for v in report.violations
        ]
        st.dataframe(pd.DataFrame(violations_data), use_container_width=True)
        st.caption(
            "📌 **Nota metodológica**: Los valores ausentes (missing) no se confunden con valores "
            "inválidos y se supervisan en la pestaña de completitud."
        )


def _build_visual_rules_editor(
    df: pd.DataFrame,
    inferences: dict,
    active_yaml_str: Optional[str],
) -> Tuple[Optional[RulesConfig], Optional[str]]:
    """Renders the visual (no-code) rule builder and returns the resulting RulesConfig and YAML.

    Returns:
        Tuple of (RulesConfig | None, yaml_string | None).
        Both are None if the user hasn't confirmed any rules yet.
    """
    st.markdown(
        "> Construye reglas de validación seleccionando variables y ajustando sus restricciones. "
        "No es necesario conocer el formato YAML."
    )

    # Pre-populate from active rules so the editor starts with existing constraints
    preloaded: Dict[str, ColumnRuleConfig] = {}
    if active_yaml_str:
        try:
            preloaded = load_rules_from_yaml(active_yaml_str).columns
        except Exception:
            pass

    # Classify columns into constrainable / non-constrainable
    col_options: List[str] = []
    col_hint: Dict[str, str] = {}  # column_name -> display hint

    for col_name in df.columns:
        inf = inferences.get(str(col_name))
        if inf is None:
            continue
        ct = _infer_constraint_type(inf.semantic_type)
        if ct is None:
            col_hint[str(col_name)] = f"🔑 Identificador — no aplica restricción de rango/categoría"
        else:
            icon = "🔢" if ct == ConstraintType.NUMERIC else "🏷️"
            col_options.append(str(col_name))
            col_hint[str(col_name)] = (
                f"{icon} {inf.semantic_type.value}  •  confianza {inf.confidence.value} ({inf.score:.0%})"
            )

    if not col_options:
        st.warning("No hay variables numéricas ni categóricas en el dataset para configurar reglas.")
        return None, None

    # Column multi-selector
    default_selection = [c for c in preloaded if c in col_options]
    selected_cols: List[str] = st.multiselect(
        "Selecciona las variables a las que quieres aplicar reglas:",
        options=col_options,
        default=default_selection,
        format_func=lambda c: f"{c}  —  {col_hint.get(c, '')}",
        help="Sólo aparecen variables numéricas y categóricas. Los identificadores se excluyen automáticamente.",
    )

    if not selected_cols:
        st.info("👆 Selecciona al menos una variable para empezar a configurar reglas.")
        return None, None

    st.markdown("---")
    built_columns: Dict[str, ColumnRuleConfig] = {}

    for col_name in selected_cols:
        inf = inferences.get(col_name)
        ct = _infer_constraint_type(inf.semantic_type) if inf else None
        existing = preloaded.get(col_name)

        with st.expander(
            f"**{col_name}** — {col_hint.get(col_name, '')}",
            expanded=True,
        ):
            if ct == ConstraintType.NUMERIC:
                # Determine sensible defaults from data or pre-loaded rules
                series = pd.to_numeric(df[col_name], errors="coerce").dropna()
                data_min = float(series.min()) if not series.empty else 0.0
                data_max = float(series.max()) if not series.empty else 100.0

                default_min = (
                    existing.numeric.min_value
                    if existing and existing.numeric and existing.numeric.min_value is not None
                    else data_min
                )
                default_max = (
                    existing.numeric.max_value
                    if existing and existing.numeric and existing.numeric.max_value is not None
                    else data_max
                )

                st.caption(
                    f"Rango observado en el dataset: **{data_min:g}** → **{data_max:g}**"
                )

                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    use_min = st.checkbox(
                        "Activar valor mínimo permitido",
                        value=existing is not None and existing.numeric is not None and existing.numeric.min_value is not None,
                        key=f"use_min_{col_name}",
                    )
                    min_val: Optional[float] = None
                    if use_min:
                        min_val = st.number_input(
                            "Mínimo",
                            value=float(default_min),
                            key=f"min_{col_name}",
                            help=f"Cualquier valor por debajo de este límite se marcará como infracción.",
                        )

                with r_col2:
                    use_max = st.checkbox(
                        "Activar valor máximo permitido",
                        value=existing is not None and existing.numeric is not None and existing.numeric.max_value is not None,
                        key=f"use_max_{col_name}",
                    )
                    max_val: Optional[float] = None
                    if use_max:
                        max_val = st.number_input(
                            "Máximo",
                            value=float(default_max),
                            key=f"max_{col_name}",
                            help=f"Cualquier valor por encima de este límite se marcará como infracción.",
                        )

                if use_min and use_max and min_val is not None and max_val is not None and min_val > max_val:
                    st.error("❌ El mínimo no puede ser mayor que el máximo.")
                    continue

                if use_min or use_max:
                    built_columns[col_name] = ColumnRuleConfig(
                        column_name=col_name,
                        rule_type=ConstraintType.NUMERIC,
                        numeric=NumericRuleConfig(min_value=min_val, max_value=max_val),
                    )
                else:
                    st.warning("⚠️ Activa al menos mínimo o máximo para que la regla sea efectiva.")

            elif ct == ConstraintType.CATEGORICAL:
                # Collect observed categories from the actual data
                observed = sorted(
                    [str(v) for v in df[col_name].dropna().unique()],
                    key=lambda x: x.lower(),
                )

                existing_allowed = (
                    [str(v) for v in existing.categorical.allowed]
                    if existing and existing.categorical
                    else observed
                )

                st.caption(
                    f"Categorías observadas en el dataset: "
                    + ", ".join(f"`{v}`" for v in observed)
                )

                allowed_vals = st.multiselect(
                    "Valores permitidos (marca los que son válidos):",
                    options=observed,
                    default=[v for v in existing_allowed if v in observed],
                    key=f"allowed_{col_name}",
                    help=(
                        "Cualquier valor que NO esté en esta lista se considerará una infracción. "
                        "Los valores ausentes (NaN) nunca se marcan como infracciones."
                    ),
                )

                # Allow adding custom values not seen in the data
                custom_raw = st.text_input(
                    "Añadir valores permitidos adicionales (separados por coma):",
                    value="",
                    key=f"custom_{col_name}",
                    help="Útil si el dataset de muestra no contiene todas las categorías válidas.",
                )
                custom_vals = [v.strip() for v in custom_raw.split(",") if v.strip()]
                all_allowed = list(dict.fromkeys(allowed_vals + custom_vals))  # preserve order, deduplicate

                if all_allowed:
                    built_columns[col_name] = ColumnRuleConfig(
                        column_name=col_name,
                        rule_type=ConstraintType.CATEGORICAL,
                        categorical=CategoricalRuleConfig(allowed=all_allowed),
                    )
                else:
                    st.warning("⚠️ Selecciona al menos un valor permitido para que la regla sea efectiva.")

    if not built_columns:
        return None, None

    resulting_config = RulesConfig(columns=built_columns)
    resulting_yaml = _rules_config_to_yaml_str(resulting_config)
    return resulting_config, resulting_yaml


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Clinical Data Audit",
        page_icon="🏥",
        layout="wide",
    )

    st.title("🏥 Clinical Data Audit")
    st.caption(
        "Auditoría, control de calidad y validación de reglas clínicas en datasets tabulares."
    )

    # ── Sidebar: Ingesta ──────────────────────────────────────────────────────
    st.sidebar.header("📁 Ingesta de Datos")
    use_sample = False
    if DEFAULT_SAMPLE_DATA_PATH.exists():
        use_sample = st.sidebar.checkbox(
            "Cargar dataset clínico de muestra (data/clinical_sample.csv)",
            value=False,
            help="Utiliza el dataset de prueba incluido en el repositorio para evaluación rápida.",
        )

    uploaded_file = st.sidebar.file_uploader(
        "O sube tu propio archivo CSV",
        type=["csv"],
        help="Carga un archivo CSV clínico para analizar.",
    )

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Reglas de Validación")
    uploaded_rules = st.sidebar.file_uploader(
        "Sube tu archivo de reglas YAML",
        type=["yaml", "yml"],
        help=(
            "Sube un archivo YAML de reglas personalizado. "
            "Si se omite, se usará config/clinical_rules.yaml. "
            "Puedes descargarlo desde la pestaña de auditoría, modificarlo y volver a subirlo aquí."
        ),
    )

    # ── Determinar fuente de datos ────────────────────────────────────────────
    df: Optional[pd.DataFrame] = None
    data_label: str = ""

    if uploaded_file is not None:
        try:
            df = load_clinical_data(uploaded_file)
            data_label = uploaded_file.name
        except Exception as exc:
            st.error(f"Error al cargar el archivo subido: {exc}")
            return
    elif use_sample and DEFAULT_SAMPLE_DATA_PATH.exists():
        try:
            df = load_clinical_data(DEFAULT_SAMPLE_DATA_PATH)
            data_label = "clinical_sample.csv"
        except Exception as exc:
            st.error(f"Error al cargar el dataset de muestra: {exc}")
            return

    if df is None:
        st.info(
            "👈 Comienza seleccionando el dataset de muestra o subiendo un archivo CSV en la barra lateral."
        )
        return

    # ── Determinar reglas activas ─────────────────────────────────────────────
    # Priority: uploaded YAML > default file
    active_yaml_str: Optional[str] = None
    rules_source_label: str = ""

    if uploaded_rules is not None:
        try:
            active_yaml_str = uploaded_rules.getvalue().decode("utf-8")
            load_rules_from_yaml(active_yaml_str)  # validate syntax
            rules_source_label = f"📎 {uploaded_rules.name} (subido)"
            st.sidebar.success(f"Reglas YAML cargadas: `{uploaded_rules.name}`")
        except Exception as exc:
            st.sidebar.error(f"Error al parsear el YAML subido: {exc}")
            active_yaml_str = None
    elif DEFAULT_RULES_PATH.exists():
        try:
            active_yaml_str = DEFAULT_RULES_PATH.read_text(encoding="utf-8")
            load_rules_from_yaml(active_yaml_str)  # validate
            rules_source_label = "📄 config/clinical_rules.yaml (por defecto)"
            st.sidebar.info("Utilizando reglas de `config/clinical_rules.yaml`.")
        except Exception as exc:
            st.sidebar.error(f"Error al cargar reglas por defecto: {exc}")
            active_yaml_str = None

    # ── Análisis base (siempre ejecutado con el dataset cargado) ──────────────
    quality_report = generate_quality_report(df, file_name=data_label)
    inferences = infer_dataset_types(df)

    # Validation with active YAML rules (may be overridden by visual builder in tab2)
    rules_config_from_yaml: Optional[RulesConfig] = None
    if active_yaml_str:
        try:
            rules_config_from_yaml = load_rules_from_yaml(active_yaml_str)
        except Exception:
            pass

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_profile, tab_audit = st.tabs(
        [
            "📊 1. Perfil Descriptivo y Completitud",
            "🚨 2. Auditoría de Reglas Clínicas",
        ]
    )

    # ═════════════════════════════════════════════════════════════════════════
    # PESTAÑA 1: PERFIL DESCRIPTIVO Y COMPLETITUD
    # ═════════════════════════════════════════════════════════════════════════
    with tab_profile:
        st.markdown(
            "> **Perfilado Estructural y Semántico**: Describe dimensiones, completitud e infiere "
            "automáticamente el rol bioestadístico de cada variable mediante heurísticas de dominio."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Dataset", quality_report.file_name or "En memoria")
        col2.metric("Total Registros", f"{quality_report.row_count:,}")
        col3.metric("Variables (Columnas)", quality_report.column_count)
        col4.metric("Duplicados Exactos", quality_report.exact_duplicates_count)

        st.subheader("Calidad y Clasificación Semántica por Variable")
        profile_rows = []
        for col in quality_report.columns.values():
            inf = inferences.get(col.name)
            sem_type = inf.semantic_type.value if inf else "-"
            conf_level = f"{inf.confidence.value} ({inf.score:.0%})" if inf else "-"

            profile_rows.append(
                {
                    "Variable": col.name,
                    "Tipo Inferido": sem_type,
                    "Confianza": conf_level,
                    "Tipo Pandas": col.dtype,
                    "Valores Faltantes": col.missing_count,
                    "% Faltantes": f"{col.missing_percentage:.1f}%",
                    "Valores Únicos": col.unique_count,
                    "Completitud": (
                        "✅ Completo" if col.missing_count == 0 else f"⚠️ {col.missing_percentage:.1f}% Nulos"
                    ),
                }
            )
        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True)

        with st.expander("🧠 Justificación de la Inferencia Heurística por Variable", expanded=False):
            for col_name, inf in inferences.items():
                st.markdown(
                    f"- **`{col_name}`** $\\rightarrow$ **{inf.semantic_type.value}** "
                    f"*(Confianza: `{inf.confidence.value}`, Score: `{inf.score:.2f}`)*: "
                    f"{inf.rationale}"
                )

        with st.expander("🔍 Vista previa del dataset (primeros registros)", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # PESTAÑA 2: AUDITORÍA DE REGLAS CLÍNICAS
    # ═════════════════════════════════════════════════════════════════════════
    with tab_audit:
        st.markdown(
            "> **Auditoría de Calidad**: Evalúa los datos contra restricciones clínicas "
            "configuradas (rangos fisiológicos y categorías permitidas). Las infracciones "
            "representan **datos inválidos o imposibles**, no valores ausentes."
        )

        # ── Sub-tabs dentro de la pestaña de auditoría ────────────────────────
        sub_results, sub_yaml_editor, sub_visual_editor = st.tabs(
            [
                "📋 Resultados de Auditoría",
                "🛠️ Editor YAML  (usuario técnico)",
                "🎛️ Constructor Visual  (sin código)",
            ]
        )

        # ─────────────────────────────────────────────────────────────────────
        # SUB-TAB A: RESULTADOS DE AUDITORÍA
        # ─────────────────────────────────────────────────────────────────────
        with sub_results:
            if rules_config_from_yaml is None:
                st.warning(
                    "No hay reglas de validación activas. "
                    "Sube un archivo YAML desde la barra lateral, o configura reglas en las pestañas de editor."
                )
            else:
                if rules_source_label:
                    st.caption(f"Fuente de reglas activa: {rules_source_label}")
                report = validate_dataframe(df, rules_config_from_yaml)
                _render_violations_table(report)

        # ─────────────────────────────────────────────────────────────────────
        # SUB-TAB B: EDITOR YAML (USUARIO TÉCNICO)
        # ─────────────────────────────────────────────────────────────────────
        with sub_yaml_editor:
            st.markdown(
                "Aquí puedes ver las reglas activas en formato YAML, **descargarlas**, "
                "modificarlas con cualquier editor de texto y **volver a subirlas** desde la barra lateral."
            )

            # Determine YAML to display: uploaded > default > empty template
            if active_yaml_str:
                display_yaml = active_yaml_str
                yaml_filename = (
                    uploaded_rules.name if uploaded_rules else "clinical_rules.yaml"
                )
            else:
                display_yaml = (
                    "# No hay reglas activas. Ejemplo de estructura YAML:\n"
                    "# nombre_columna:\n"
                    "#   type: numeric      # o categorical\n"
                    "#   min: 0             # para numeric\n"
                    "#   max: 120           # para numeric\n"
                    "#   allowed:           # para categorical\n"
                    "#     - ValorA\n"
                    "#     - ValorB\n"
                )
                yaml_filename = "clinical_rules_ejemplo.yaml"

            st.code(display_yaml, language="yaml")

            st.download_button(
                label="⬇️ Descargar reglas YAML",
                data=display_yaml.encode("utf-8"),
                file_name=yaml_filename,
                mime="text/yaml",
                help=(
                    "Descarga el archivo YAML con las reglas actuales. "
                    "Edítalo con cualquier editor de texto y súbelo de nuevo desde la barra lateral."
                ),
            )

            st.info(
                "💡 **Flujo de trabajo**: descarga → edita con un editor de texto → "
                "sube el archivo modificado en **⚙️ Reglas de Validación** (barra lateral izquierda)."
            )

            # Reference card
            with st.expander("📖 Referencia rápida del formato YAML", expanded=False):
                st.markdown(
                    """
**Regla numérica** — para variables como edad, peso, tensión arterial:
```yaml
nombre_columna:
  type: numeric
  min: 0      # opcional — mínimo permitido
  max: 120    # opcional — máximo permitido
```

**Regla categórica** — para variables como sexo, diagnóstico, grupo:
```yaml
nombre_columna:
  type: categorical
  allowed:
    - ValorPermitidoA
    - ValorPermitidoB
```

> Los valores ausentes (`NaN`, celdas vacías) **nunca** se cuentan como infracciones.
> Las infracciones sólo se reportan para valores presentes que incumplan la restricción.
                    """
                )

        # ─────────────────────────────────────────────────────────────────────
        # SUB-TAB C: CONSTRUCTOR VISUAL (USUARIO NO TÉCNICO)
        # ─────────────────────────────────────────────────────────────────────
        with sub_visual_editor:
            visual_config, visual_yaml = _build_visual_rules_editor(
                df=df,
                inferences=inferences,
                active_yaml_str=active_yaml_str,
            )

            if visual_config is not None and visual_yaml is not None:
                st.markdown("---")
                st.subheader("📋 Resultado de la Validación con las Reglas Configuradas")

                visual_report = validate_dataframe(df, visual_config)
                _render_violations_table(visual_report)

                st.markdown("---")
                st.subheader("📄 YAML generado por el constructor")
                st.caption(
                    "Este es el YAML equivalente a las reglas que acabas de configurar. "
                    "Puedes descargarlo para reutilizarlo o compartirlo con un usuario técnico."
                )
                st.code(visual_yaml, language="yaml")
                st.download_button(
                    label="⬇️ Descargar reglas generadas como YAML",
                    data=visual_yaml.encode("utf-8"),
                    file_name="mis_reglas_clinicas.yaml",
                    mime="text/yaml",
                    help=(
                        "Guarda estas reglas como archivo YAML para subirlas directamente "
                        "en futuras sesiones sin tener que volver a configurarlas."
                    ),
                )


if __name__ == "__main__":
    main()
