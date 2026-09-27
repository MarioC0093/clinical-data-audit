"""Streamlit UI entrypoint for Clinical Data Audit.

Presentation layer purely consuming core clinical_audit modules without
containing validation or statistical logic.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Ensure src/ is on Python search path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
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
from clinical_audit.stats.summarizer import (
    compute_grouped_freq,
    compute_stats_summary,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

_APP_DIR = Path(__file__).resolve().parent
DEFAULT_RULES_PATH = _APP_DIR / "config" / "clinical_rules.yaml"
DEFAULT_SAMPLE_DATA_PATH = _APP_DIR / "data" / "clinical_sample.csv"
APP_CONFIG_PATH = _APP_DIR / "config" / "app_config.yaml"
CSS_PATH = _APP_DIR / "static" / "styles.css"

# ── Semantic type sets ────────────────────────────────────────────────────────

_NUMERIC_SEMANTIC_TYPES = {
    SemanticType.NUMERIC_CONTINUOUS,
    SemanticType.NUMERIC_DISCRETE,
}

_CATEGORICAL_SEMANTIC_TYPES = {
    SemanticType.BINARY,
    SemanticType.CATEGORICAL_NOMINAL,
}

_FACTOR_SEMANTIC_TYPES = {
    SemanticType.BINARY,
    SemanticType.CATEGORICAL_NOMINAL,
}


# ── Config loaders ────────────────────────────────────────────────────────────

def _load_app_config() -> dict:
    """Loads app_config.yaml; returns empty dict on any error."""
    try:
        with open(APP_CONFIG_PATH, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _load_css() -> None:
    """Injects static/styles.css into the Streamlit page if the file exists."""
    if CSS_PATH.exists():
        css_text = CSS_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    """Maps a SemanticType to the most appropriate ConstraintType, or None."""
    if semantic_type in _NUMERIC_SEMANTIC_TYPES:
        return ConstraintType.NUMERIC
    if semantic_type in _CATEGORICAL_SEMANTIC_TYPES:
        return ConstraintType.CATEGORICAL
    return None


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
    """Renders the visual (no-code) rule builder and returns RulesConfig + YAML."""
    st.markdown(
        "> Construye reglas de validación seleccionando variables y ajustando sus restricciones. "
        "No es necesario conocer el formato YAML."
    )

    preloaded: Dict[str, ColumnRuleConfig] = {}
    if active_yaml_str:
        try:
            preloaded = load_rules_from_yaml(active_yaml_str).columns
        except Exception:
            pass

    col_options: List[str] = []
    col_hint: Dict[str, str] = {}

    for col_name in df.columns:
        inf = inferences.get(str(col_name))
        if inf is None:
            continue
        ct = _infer_constraint_type(inf.semantic_type)
        if ct is None:
            col_hint[str(col_name)] = "🔑 Identificador — no aplica restricción de rango/categoría"
        else:
            icon = "🔢" if ct == ConstraintType.NUMERIC else "🏷️"
            col_options.append(str(col_name))
            col_hint[str(col_name)] = (
                f"{icon} {inf.semantic_type.value}  •  confianza {inf.confidence.value} ({inf.score:.0%})"
            )

    if not col_options:
        st.warning("No hay variables numéricas ni categóricas en el dataset para configurar reglas.")
        return None, None

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
                    "Categorías observadas en el dataset: "
                    + ", ".join(f"`{v}`" for v in observed)
                )
                allowed_vals = st.multiselect(
                    "Valores permitidos (marca los que son válidos):",
                    options=observed,
                    default=[v for v in existing_allowed if v in observed],
                    key=f"allowed_{col_name}",
                )
                custom_raw = st.text_input(
                    "Añadir valores permitidos adicionales (separados por coma):",
                    value="",
                    key=f"custom_{col_name}",
                )
                custom_vals = [v.strip() for v in custom_raw.split(",") if v.strip()]
                all_allowed = list(dict.fromkeys(allowed_vals + custom_vals))

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


# ── Tab 3: Statistical Summary ────────────────────────────────────────────────

def _render_stats_tab(df: pd.DataFrame, inferences: dict) -> None:
    """Renders the complete statistical summary tab (Tab 3)."""

    st.markdown(
        "> **Resumen Estadístico**: estadísticos descriptivos por variable, tablas de frecuencias "
        "y visualizaciones exploratorias de variables categóricas/factor."
    )

    stats = compute_stats_summary(df)

    # ── 3.0 Dataset-level KPIs ────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Registros", f"{stats.n_rows:,}")
    k2.metric("Variables", stats.n_cols)
    k3.metric("Filas completas", f"{stats.n_complete_rows:,}")
    k4.metric("% Completitud", f"{stats.complete_row_pct:.1f}%")
    k5.metric("Vars. numéricas", stats.n_numeric_cols)
    k6.metric("Vars. categóricas", stats.n_categorical_cols)

    st.markdown("---")

    # ── 3.1 Numeric descriptive statistics ───────────────────────────────────
    st.subheader("3.1 Estadísticos descriptivos (variables numéricas)")

    if not stats.numeric_stats:
        st.info("No se encontraron variables numéricas en este dataset.")
    else:
        desc_rows = [
            {
                "Variable": s.column,
                "N válido": s.count,
                "Faltantes": s.missing,
                "% Faltantes": f"{s.missing_pct:.1f}%",
                "Media": s.mean,
                "Desv. típica": s.std,
                "Mediana": s.median,
                "Q1": s.q1,
                "Q3": s.q3,
                "IQR": s.iqr,
                "Mín.": s.min,
                "Máx.": s.max,
                "Asimetría": s.skewness,
                "Curtosis": s.kurtosis,
                "CV (%)": s.cv,
            }
            for s in stats.numeric_stats
        ]
        st.dataframe(
            pd.DataFrame(desc_rows).set_index("Variable"),
            use_container_width=True,
        )

        with st.expander("ℹ️ Descripción de los estadísticos"):
            st.markdown(
                """
| Estadístico | Descripción |
|---|---|
| **N válido** | Número de registros sin valor ausente |
| **Faltantes / %** | Valores ausentes absolutos y relativos |
| **Media** | Media aritmética |
| **Desv. típica** | Desviación estándar muestral |
| **Mediana** | Percentil 50 (Q2) |
| **Q1 / Q3** | Percentiles 25 y 75 |
| **IQR** | Rango intercuartílico (Q3 − Q1) |
| **Mín. / Máx.** | Valores extremos observados |
| **Asimetría** | Sesgo de la distribución (>0 cola derecha) |
| **Curtosis** | Apuntamiento relativo a distribución normal |
| **CV (%)** | Coeficiente de variación = (σ / μ) × 100 |
"""
            )

    st.markdown("---")

    # ── 3.2 Frequency tables ──────────────────────────────────────────────────
    st.subheader("3.2 Tablas de frecuencias (variables categóricas / factor)")

    factor_cols = list(stats.freq_tables.keys())

    if not factor_cols:
        st.info("No se encontraron variables categóricas en este dataset.")
    else:
        with st.expander("📊 Frecuencias simples por variable", expanded=True):
            col_sel = st.selectbox(
                "Selecciona una variable para ver su distribución:",
                options=factor_cols,
                key="freq_col_selector",
            )
            if col_sel:
                freq_data = stats.freq_tables[col_sel]
                freq_df = pd.DataFrame(
                    [
                        {"Categoría": f.category, "Recuento": f.count, "Porcentaje (%)": f.pct}
                        for f in freq_data
                    ]
                )
                st.dataframe(freq_df, use_container_width=True, hide_index=True)

        with st.expander("📊 Frecuencias por datos agrupados (dos variables)", expanded=True):
            st.markdown(
                "Selecciona dos variables categóricas: **variable de agrupación** (filas) "
                "y **variable de valor** (distribución interna). "
                "Si la variable de valor es **binaria**, se muestra la proporción de la categoría positiva; "
                "si no, se muestran recuentos de cada categoría."
            )

            g_col1, g_col2 = st.columns(2)
            with g_col1:
                group_var = st.selectbox(
                    "Variable de agrupación (eje de filas):",
                    options=factor_cols,
                    key="grouped_group_col",
                )
            with g_col2:
                remaining = [c for c in factor_cols if c != group_var]
                if not remaining:
                    st.warning("Se necesitan al menos dos variables categóricas.")
                    value_var = None
                else:
                    value_var = st.selectbox(
                        "Variable de valor (distribución interna):",
                        options=remaining,
                        key="grouped_value_col",
                    )

            if group_var and value_var:
                grouped_rows = compute_grouped_freq(df, group_col=group_var, value_col=value_var)
                is_bin = df[value_var].nunique(dropna=True) == 2

                if grouped_rows:
                    if is_bin:
                        # Binary: one row per group, show proportion
                        bin_df = pd.DataFrame(
                            [
                                {
                                    group_var: r.group_label,
                                    f"N ({r.subgroup_label})": r.count,
                                    "Proporción (%)": r.pct,
                                }
                                for r in grouped_rows
                            ]
                        )
                        st.dataframe(bin_df, use_container_width=True, hide_index=True)
                    else:
                        # Non-binary: pivot for readability
                        pivot_df = pd.DataFrame(
                            [
                                {
                                    "Grupo": r.group_label,
                                    "Categoría": r.subgroup_label,
                                    "Recuento": r.count,
                                    "Porcentaje dentro del grupo (%)": r.pct,
                                }
                                for r in grouped_rows
                            ]
                        )
                        st.dataframe(pivot_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No se encontraron registros con ambas variables presentes.")

    st.markdown("---")

    # ── 3.3 Visualisations ────────────────────────────────────────────────────
    st.subheader("3.3 Visualización de variables factor")

    # Detect factor columns via inference (BINARY or CATEGORICAL_NOMINAL)
    factor_inferred = [
        col
        for col, inf in inferences.items()
        if inf.semantic_type in _FACTOR_SEMANTIC_TYPES
        and col in df.columns
        and df[col].nunique(dropna=True) >= 2
    ]

    if len(factor_inferred) < 1:
        st.info("No se detectaron variables de tipo factor en este dataset.")
        return

    st.markdown(
        "Selecciona **dos variables de tipo factor** (binaria, nominal o categórica) "
        "para el gráfico exploratorio."
    )

    v_col1, v_col2, v_col3 = st.columns([2, 2, 2])
    with v_col1:
        var1 = st.selectbox(
            "Variable 1 (primer nivel / agrupación):",
            options=factor_inferred,
            key="plot_var1",
        )
    with v_col2:
        var2_opts = [c for c in factor_inferred if c != var1]
        if not var2_opts:
            st.warning("Se necesitan al menos dos variables factor.")
            return
        var2 = st.selectbox(
            "Variable 2 (segundo nivel / color):",
            options=var2_opts,
            key="plot_var2",
        )
    with v_col3:
        chart_type = st.selectbox(
            "Tipo de gráfico:",
            options=["Gráfico de sectores (Pie)", "Gráfico de barras agrupadas"],
            key="plot_chart_type",
        )

    plot_df = df[[var1, var2]].dropna()

    if plot_df.empty:
        st.warning("No hay datos disponibles para las variables seleccionadas.")
        return

    if chart_type == "Gráfico de sectores (Pie)":
        # Two pie charts side-by-side: one per category of var1
        categories_var1 = sorted(plot_df[var1].unique().tolist(), key=str)
        n_cats = len(categories_var1)
        cols = st.columns(min(n_cats, 3))

        for i, cat in enumerate(categories_var1):
            subset = plot_df[plot_df[var1] == cat][var2].value_counts()
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(
                subset.values,
                labels=subset.index.tolist(),
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.set_title(f"{var1} = {cat}", fontsize=11, fontweight="bold")
            with cols[i % len(cols)]:
                st.pyplot(fig)
            plt.close(fig)

    else:  # Gráfico de barras agrupadas
        fig = px.histogram(
            plot_df,
            x=var1,
            color=var2,
            barmode="group",
            title=f"Distribución de {var2} por {var1}",
            labels={var1: var1, "count": "Recuento", var2: var2},
            text_auto=True,
        )
        fig.update_layout(
            xaxis_title=var1,
            yaxis_title="Recuento",
            legend_title=var2,
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Sidebar authorship ────────────────────────────────────────────────────────

def _render_sidebar_authorship(cfg: dict) -> None:
    """Renders the authorship and contact block at the bottom of the sidebar."""
    auth = cfg.get("authorship", {})
    contact = cfg.get("contact", {})

    team = auth.get("team", "")
    institution = auth.get("institution", "")
    year = auth.get("year", "")
    email = contact.get("email", "")
    repo = contact.get("repository", "")
    linkedin = contact.get("linkedin", "")
    docs = contact.get("docs", "")

    # HTML for SVG icons
    icon_email = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-envelope-fill" viewBox="0 0 16 16"><path d="M.05 3.555A2 2 0 0 1 2 2h12a2 2 0 0 1 1.95 1.555L8 8.414zM0 4.697v7.104l5.803-3.558zM6.761 8.83l-6.57 4.027A2 2 0 0 0 2 14h12a2 2 0 0 0 1.808-1.144l-6.57-4.027L8 9.586zm3.436-.232L16 11.801V4.697z"/></svg>'
    icon_github = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-github" viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8"/></svg>'
    icon_linkedin = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-linkedin" viewBox="0 0 16 16"><path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854zm4.943 12.248V6.169H2.542v7.225zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248S2.4 3.226 2.4 3.934c0 .694.521 1.248 1.327 1.248zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016l.016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225z"/></svg>'
    icon_docs = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-book-half" viewBox="0 0 16 16"><path d="M8.5 2.687c.654-.689 1.782-.886 3.112-.752 1.234.124 2.503.523 3.388.893v9.923c-.918-.35-2.107-.692-3.287-.81-1.094-.111-2.278-.039-3.213.492zM8 1.783C7.015.936 5.587.81 4.287.94c-1.514.153-3.042.672-3.994 1.105A.5.5 0 0 0 0 2.5v11a.5.5 0 0 0 .707.455c.882-.4 2.303-.881 3.68-1.02 1.409-.142 2.59.087 3.223.877a.5.5 0 0 0 .78 0c.633-.79 1.814-1.019 3.222-.877 1.378.139 2.8.62 3.681 1.02A.5.5 0 0 0 16 13.5v-11a.5.5 0 0 0-.293-.455c-.952-.433-2.48-.952-3.994-1.105C10.413.809 8.985.936 8 1.783"/></svg>'

    lines: List[str] = []
    if team:
        lines.append(f"<strong>{team}</strong>")
    if institution:
        lines.append(institution)
    if year:
        lines.append(f"&copy; {year}")

    contact_parts: List[str] = []
    if email:
        email_clean = email.replace("mailto:", "")
        contact_parts.append(f"<a href='mailto:{email_clean}' title='{email_clean}' target='_blank' style='margin-right: 12px;'>{icon_email}</a>")
    if linkedin:
        contact_parts.append(f"<a href='{linkedin}' title='LinkedIn' target='_blank' style='margin-right: 12px;'>{icon_linkedin}</a>")
    if repo:
        contact_parts.append(f"<a href='{repo}' title='Repositorio GitHub' target='_blank' style='margin-right: 12px;'>{icon_github}</a>")
    if docs:
        contact_parts.append(f"<a href='{docs}' title='Documentación' target='_blank'>{icon_docs}</a>")

    html = "<div class='sidebar-authorship'>"
    if lines:
        html += "<br>".join(lines) + "<br><br>"
    if contact_parts:
        html += "<div style='display: flex; align-items: center;'>" + "".join(contact_parts) + "</div>"
    html += "</div>"

    st.sidebar.markdown("---")
    st.sidebar.markdown(html, unsafe_allow_html=True)



# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    app_cfg = _load_app_config()
    app_section = app_cfg.get("app", {})

    st.set_page_config(
        page_title=app_section.get("title", "Clinical Data Audit"),
        page_icon=app_section.get("icon", "🏥"),
        layout="wide",
    )

    # Inject external CSS
    _load_css()

    st.title(f"{app_section.get('icon', '🏥')} {app_section.get('title', 'Clinical Data Audit')}")
    st.caption(
        app_section.get(
            "subtitle",
            "Auditoría, control de calidad y validación de reglas clínicas en datasets tabulares.",
        )
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

    # Authorship block at the bottom of the sidebar
    _render_sidebar_authorship(app_cfg)

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
    active_yaml_str: Optional[str] = None
    rules_source_label: str = ""

    if uploaded_rules is not None:
        try:
            active_yaml_str = uploaded_rules.getvalue().decode("utf-8")
            load_rules_from_yaml(active_yaml_str)
            rules_source_label = f"📎 {uploaded_rules.name} (subido)"
            st.sidebar.success(f"Reglas YAML cargadas: `{uploaded_rules.name}`")
        except Exception as exc:
            st.sidebar.error(f"Error al parsear el YAML subido: {exc}")
            active_yaml_str = None
    elif DEFAULT_RULES_PATH.exists():
        try:
            active_yaml_str = DEFAULT_RULES_PATH.read_text(encoding="utf-8")
            load_rules_from_yaml(active_yaml_str)
            rules_source_label = "📄 config/clinical_rules.yaml (por defecto)"
            st.sidebar.info("Utilizando reglas de `config/clinical_rules.yaml`.")
        except Exception as exc:
            st.sidebar.error(f"Error al cargar reglas por defecto: {exc}")
            active_yaml_str = None

    # ── Análisis base ─────────────────────────────────────────────────────────
    quality_report = generate_quality_report(df, file_name=data_label)
    inferences = infer_dataset_types(df)

    rules_config_from_yaml: Optional[RulesConfig] = None
    if active_yaml_str:
        try:
            rules_config_from_yaml = load_rules_from_yaml(active_yaml_str)
        except Exception:
            pass

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_profile, tab_audit, tab_stats = st.tabs(
        [
            "📊 1. Perfil Descriptivo y Completitud",
            "🚨 2. Auditoría de Reglas Clínicas",
            "📈 3. Resumen Estadístico",
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

        sub_results, sub_yaml_editor, sub_visual_editor = st.tabs(
            [
                "📋 Resultados de Auditoría",
                "🛠️ Editor YAML  (usuario técnico)",
                "🎛️ Constructor Visual  (sin código)",
            ]
        )

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

        with sub_yaml_editor:
            st.markdown(
                "Aquí puedes ver las reglas activas en formato YAML, **descargarlas**, "
                "modificarlas con cualquier editor de texto y **volver a subirlas** desde la barra lateral."
            )

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
            )

            st.info(
                "💡 **Flujo de trabajo**: descarga → edita con un editor de texto → "
                "sube el archivo modificado en **⚙️ Reglas de Validación** (barra lateral izquierda)."
            )

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
                )

    # ═════════════════════════════════════════════════════════════════════════
    # PESTAÑA 3: RESUMEN ESTADÍSTICO
    # ═════════════════════════════════════════════════════════════════════════
    with tab_stats:
        _render_stats_tab(df, inferences)


if __name__ == "__main__":
    main()
