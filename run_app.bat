@echo off
setlocal EnableDelayedExpansion

echo.
echo ======================================================
echo   Clinical Data Audit -- Lanzador local
echo ======================================================
echo.

REM Cambiamos al directorio donde esta este .bat (raiz del proyecto)
pushd "%~dp0"

REM ── 1. Verificar que Python está disponible ─────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado en PATH.
    echo         Instala Python 3.12 o superior desde https://www.python.org/downloads/
    echo         y asegurate de marcar "Add Python to PATH" durante la instalacion.
    pause
    exit /b 1
)

REM ── 2. Crear entorno virtual si no existe ───────────────────────────────────
if not exist ".venv\" (
    echo [INFO] Creando entorno virtual en .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
    echo [INFO] Entorno virtual creado.
    echo [INFO] Actualizando pip ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
) else (
    echo [INFO] Entorno virtual existente detectado en .venv\
)

REM ── 3. Sincronizar dependencias siempre (por si requirements.txt cambió) ────
echo [INFO] Verificando dependencias desde requirements.txt ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Error al instalar dependencias.
    pause
    exit /b 1
)
echo [INFO] Dependencias OK.

REM ── 4. Lanzar la aplicacion Streamlit ────────────────────────────────────────
echo.
echo [INFO] Iniciando Clinical Data Audit en el navegador...
echo [INFO] Para detener la aplicacion, cierra esta ventana o pulsa Ctrl+C
echo.

".venv\Scripts\python.exe" -m streamlit run app.py

timeout /t 500

popd
endlocal