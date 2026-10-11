# TC5061 · Código de experimentación base con MLflow

Alan Alfonso Boldo Reyes (A01840556) · Operaciones de aprendizaje automático (TC5061.10) · Maestría en Inteligencia Artificial Aplicada, Tec de Monterrey

Migración del baseline del notebook de la semana 3 (clasificación de los tres cultivares del dataset *Wine* de UCI con `StandardScaler` + `LogisticRegression`) a un script parametrizado que registra cada corrida en MLflow Tracking. Con los valores por defecto, `train.py` reproduce las métricas del baseline de ese notebook: F1 macro de validación cruzada 0.9796 ± 0.0305; exactitud 0.9722 y F1 macro 0.9710 en test.

## Contenido

| Archivo | Para qué |
|---|---|
| `train.py` | Script de entrenamiento: carga y valida los datos, separa train/test, valida cruzado, entrena, evalúa y registra en MLflow. |
| `run_experiments.py` | Lote de 10 configuraciones definidas de antemano más la repetición del baseline; compara y elige por validación cruzada. |
| `notebooks/01_eda_wine.ipynb` | Exploración de datos (solo EDA); reutiliza las funciones de `train.py`. |
| `requirements.txt` | Dependencias con versión fija; es la vía de instalación. |
| `requirements-lock.txt` | `pip freeze` del entorno (Windows 11, Python 3.13.7) con que se generaron las corridas reportadas. |
| `data/` | `wine.data` (dataset original de UCI) y `wine.names` (su descripción). |
| `reports/` | Evidencia ligera del lote reportado (`comparison.csv`, `selection.json`, `reproducibility.json`, `commands.json`); la base de MLflow y sus artefactos quedan fuera del repositorio y se regeneran con el lote. |
| `tests/` | Pruebas con `unittest`. |

## Instalación (Python 3.13)

Windows (PowerShell):

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS o Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Sin activar el entorno, los mismos comandos funcionan llamando al intérprete por su ruta (`.venv\Scripts\python.exe` en Windows, `.venv/bin/python` en macOS o Linux). En Windows conviene definir `PYTHONUTF8=1`.

El notebook de exploración necesita además Jupyter (`python -m pip install notebook nbconvert`); el script no lo usa.

## Ejecución

Desde la carpeta del repositorio y con el entorno activado.

Una corrida con los valores por defecto (la configuración del baseline):

```bash
python train.py
```

El mismo comando con todos los argumentos explícitos:

```bash
python train.py --C 1 --l1-ratio 0 --solver lbfgs --max-iter 1000 --tol 0.0001 --seed 42 --test-size 0.2 --cv-folds 5 --experiment TC5061_Experimentacion_BoldoAlan_Wine --output outputs
```

Cada corrida se agrega al experimento de la carpeta `--output`. Las 11 corridas reportadas (10 configuraciones y la repetición del baseline) salen de un lote en una carpeta limpia; tarda un par de minutos:

```bash
python run_experiments.py --output outputs
```

Interfaz de MLflow para revisar y comparar las corridas en http://127.0.0.1:5000 (se lanza desde la carpeta del repositorio porque la ruta de la base es relativa):

```bash
python -m mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

En Windows, con el valor por defecto (4 procesos), algunos procesos del servidor de MLflow 3.16.1 pueden fallar al arrancar (`OSError: [WinError 10022]`) y reiniciarse: la interfaz abre, pero la consola muestra trazas; `--workers 1` lo evita y basta para uso local.

`python train.py --help` muestra todos los argumentos con su ayuda y su valor por defecto. Para pruebas sueltas conviene otra carpeta (por ejemplo `--output outputs_pruebas`), así el experimento del lote conserva solo sus 11 corridas.

## Parámetros

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--C` | 1.0 | Inverso de la fuerza de regularización (menor = más regularización). |
| `--l1-ratio` | 0.0 | Tipo de penalización: 0 = L2, 1 = L1, intermedio = elastic-net; mayor que 0 requiere `--solver saga`. |
| `--solver` | lbfgs | Optimizador: `lbfgs`, `newton-cg` o `saga`. |
| `--max-iter` | 1000 | Máximo de iteraciones; si scikit-learn emite `ConvergenceWarning`, la corrida falla en lugar de registrar un modelo a medias. |
| `--tol` | 0.0001 | Tolerancia del criterio de paro. |
| `--seed` | 42 | Semilla de la partición, de los folds de validación cruzada y del modelo. |
| `--test-size` | 0.2 | Fracción estratificada reservada para test. |
| `--cv-folds` | 5 | Folds de validación cruzada estratificada dentro de train. |
| `--experiment` | TC5061_Experimentacion_BoldoAlan_Wine | Experimento de MLflow que agrupa las corridas. |
| `--output` | outputs | Carpeta de la base `mlflow.db`, de los artefactos (`mlartifacts/`) y de las copias locales (`runs/`, `batches/`). |

## Qué se registra en MLflow

Cada ejecución de `train.py` crea un run en el experimento `TC5061_Experimentacion_BoldoAlan_Wine`:

- **Parámetros** (`log_param`, 19): hiperparámetros, semilla, diseño de la evaluación y contexto (dataset y su SHA-256, hash del script y de `requirements.txt`).
- **Métricas** (`log_metric`, 10): `cv_f1_macro_mean` (criterio de selección) y `cv_f1_macro_std`; `test_accuracy`, `test_f1_macro` y `test_balanced_accuracy`; `train_accuracy`; el control de clase mayoritaria; `n_iter` y `n_coef_nonzero`.
- **Artefactos** (`log_artifact`): `evaluation/` (matriz de confusión PNG y CSV, reporte por clase, folds de CV, predicciones), `data/` (auditoría, partición y el dataset usado), `config/` (parámetros, comando y entorno) y `source/` (script y dependencias).
- **Modelo** (`mlflow.sklearn.log_model`): el Pipeline completo con firma y ejemplo de entrada; se recarga y se comprueba que prediga lo mismo. En MLflow 3 aparece en la sección *Logged models* del run y en la pestaña *Models* del experimento, no en el árbol de artefactos.

Al cerrar el run, el script lo relee desde MLflow y se detiene si parámetros, métricas o artefactos no coinciden con lo calculado. `run_experiments.py` etiqueta además cada run con `role`, `block` y `batch_id`, y marca el elegido con `selected = true`.

## Reproducibilidad

- Una sola semilla (`--seed`) controla la partición, los folds, el modelo y el control; el cálculo numérico corre en un hilo.
- `load_data` verifica el SHA-256 de `wine.data`; `.gitattributes` evita que Git cambie sus bytes y mantiene el resto del texto con saltos LF en cualquier sistema.
- `requirements.txt` fija las versiones y es la vía de instalación; `requirements-lock.txt` es la fotografía completa del entorno de Windows en que se generaron las corridas (incluye paquetes propios de ese sistema).
- `run_experiments.py` repite el baseline con el mismo comando y compara parámetros, métricas y hashes de las predicciones, de la partición y de los puntajes por fold (`outputs/batches/<lote>/reproducibility.json`).
- Las 11 corridas reportadas (lote `20261010-211817-d46639`) se generaron con el commit `9061d26` de este repositorio; MLflow guarda ese commit en cada run y `train.py` y `requirements.txt` no cambiaron después.

## Pruebas

```bash
python -m unittest discover -s tests -v
```

Son 16 pruebas; una de ellas ejecuta una corrida real con MLflow en una carpeta temporal.

## Datos y uso de IA

Aeberhard, S. y Forina, M. (1992). *Wine* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J (licencia CC BY 4.0).

Usé herramientas de IA (Claude y Claude Code, de Anthropic, y Codex, de OpenAI) como apoyo para refactorizar, probar, ejecutar y documentar el código; las decisiones de diseño las asumo como mías.
