# Clinical Data Audit 🏥

Herramienta modular en Python para la auditoría, validación y control de calidad de datos clínicos a partir de archivos CSV.

---

## 📐 Arquitectura del Proyecto

```text
antigravity-clinical-audit/
├── .gitignore                      # Exclusiones de Git estándar para Python
├── README.md                       # Documentación del proyecto
├── pyproject.toml                  # Configuración del paquete y pytest
├── requirements.txt                # Dependencias de producción (Streamlit Cloud)
├── requirements-dev.txt            # Dependencias de desarrollo (pytest, etc.)
├── run_app.bat                     # Lanzador local para Windows
├── app.py                          # 🚀 Punto de entrada de Streamlit
├── config/
│   └── clinical_rules.yaml         # Reglas de validación clínica por defecto
├── data/
│   └── clinical_sample.csv         # Dataset de ejemplo incluido en el repositorio
├── src/
│   └── clinical_audit/             # Paquete principal de análisis y auditoría
│       ├── __init__.py
│       ├── checks/                 # Catálogo de reglas de calidad
│       │   └── base.py
│       ├── core/                   # Modelos de dominio y contratos base
│       │   ├── models.py           # Dataclasses (QualityReport, etc.)
│       │   └── rules.py            # Clase abstracta BaseRule
│       ├── engine/                 # Motor orquestador de la auditoría
│       │   └── auditor.py          # Clase ClinicalAuditor
│       ├── inference/              # Inferencia semántica de variables
│       │   ├── classifier.py       # Función infer_dataset_types
│       │   └── models.py
│       ├── ingestion/              # Ingesta y lectura de archivos CSV
│       │   └── loader.py           # Función load_clinical_data
│       ├── quality/                # Perfilado de calidad
│       │   └── profiler.py         # Función generate_quality_report
│       ├── reporting/              # Formateo y serialización de informes
│       │   └── reporter.py
│       └── rules/                  # Motor de reglas YAML
│           ├── config.py           # Parser YAML → RulesConfig
│           ├── models.py           # Dataclasses de reglas
│           └── validator.py        # validate_dataframe
└── tests/                          # Suite de pruebas con pytest
    ├── __init__.py
    ├── conftest.py                  # Fixtures compartidas
    ├── test_inference.py
    ├── test_profiler.py
    ├── test_rules_config.py
    └── test_validator.py
```

---

## 🚀 Ejecución local (Windows)

### Opción A — Doble clic (más sencillo)

Haz doble clic en **`run_app.bat`**. El script:

1. Comprueba que Python está disponible en PATH.
2. Crea el entorno virtual `.venv/` si no existe.
3. Instala las dependencias de `requirements.txt`.
4. Lanza la aplicación Streamlit y la abre en el navegador.

> **Requisito previo**: Python 3.12 o superior instalado y en el PATH del sistema.  
> Descarga desde [python.org/downloads](https://www.python.org/downloads/) y marca *"Add Python to PATH"*.

### Opción B — Terminal (PowerShell)

```powershell
# 1. Crear entorno virtual (solo la primera vez)
python -m venv .venv

# 2. Activar el entorno
.\.venv\Scripts\Activate.ps1

# 3. Instalar dependencias de producción
pip install -r requirements.txt

# 4. Lanzar la aplicación
streamlit run app.py
```

---

## 🧪 Ejecución de Tests

```powershell
# Activar entorno e instalar dependencias de desarrollo
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

# Ejecutar la suite completa
pytest
```

---

## 🌐 Despliegue en Streamlit Community Cloud

1. Sube el repositorio a GitHub (rama `main`).
2. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
3. Haz clic en **"New app"** y configura:
   - **Repository**: `tu-usuario/antigravity-clinical-audit`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Haz clic en **"Deploy"**.

Streamlit Cloud instalará automáticamente las dependencias declaradas en `requirements.txt`.

> [!IMPORTANT]
> No es necesario ningún archivo adicional de configuración (no se necesita `Procfile`, `Dockerfile`, ni GitHub Actions).

---

## 📊 Gestión de datos

### Datos de ejemplo (`data/`)

El directorio `data/` contiene únicamente fixtures de referencia incluidas en el repositorio:

| Archivo | Descripción |
|---|---|
| `clinical_sample.csv` | Dataset de 10 registros con casos válidos e inválidos para demo y tests |

### Archivos subidos por el usuario

Los archivos CSV que el usuario sube mediante `st.file_uploader()` se procesan **exclusivamente en memoria** (como objetos `BytesIO`) y **nunca se escriben en disco**. No persisten entre sesiones ni forman parte del repositorio.

---

## ⚙️ Reglas de Validación Configurables (YAML)

El sistema permite definir restricciones de dominio clínico en archivos YAML sin modificar el código fuente.

### Ejemplo de configuración (`config/clinical_rules.yaml`)

```yaml
age:
  type: numeric
  min: 0
  max: 120

bmi:
  type: numeric
  min: 10
  max: 80

sex:
  type: categorical
  allowed:
    - F
    - M

smoker:
  type: categorical
  allowed:
    - Yes
    - No

systolic_bp:
  type: numeric
  min: 50
  max: 300
```

> [!NOTE]
> **Separación de Nulos e Infracciones**: Los valores ausentes (`NaN`, `None`) son tratados como problemas de completitud y **no** se confunden con valores fuera de rango ni categorías inválidas.

---

## 🧩 Cómo añadir nuevas reglas de calidad

1. Crea un archivo en `src/clinical_audit/checks/` (por ejemplo `completeness.py`).
2. Hereda de `BaseRule` e implementa el método `evaluate(df)`:

```python
from clinical_audit.core.rules import BaseRule
from clinical_audit.core.models import RuleResult, RuleStatus, Severity
import pandas as pd

class MissingValuesCheck(BaseRule):
    rule_id = "COMP-001"
    rule_name = "Comprobación de valores nulos"
    category = "Completitud"
    severity = Severity.WARNING

    def evaluate(self, df: pd.DataFrame) -> RuleResult:
        null_count = int(df.isnull().sum().sum())
        status = RuleStatus.PASSED if null_count == 0 else RuleStatus.WARNING
        return RuleResult(
            rule_id=self.rule_id,
            rule_name=self.rule_name,
            category=self.category,
            status=status,
            severity=self.severity,
            message=f"Se encontraron {null_count} celdas vacías.",
            affected_rows_count=null_count,
        )
```

3. Registra la regla en `ClinicalAuditor`:

```python
auditor = ClinicalAuditor(rules=[MissingValuesCheck()])
report = auditor.audit(df)
```
