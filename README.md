# Polyp Detection Meta-Analysis

Estudio comparativo de modelos de la familia **YOLO** (YOLOv5, YOLO11, …) aplicados a la **detección de pólipos en imágenes de colonoscopia**, con especial atención a cómo se degrada su rendimiento ante **cambios de distribución (domain shift)** entre el conjunto de entrenamiento y distintos conjuntos de evaluación, y a su comportamiento cuando se despliegan en **hardware embebido (NVIDIA Jetson)** bajo distintas precisiones numéricas (FP32 / FP16 / INT8).

Este repositorio constituye la base experimental de mi [Trabajo de Fin de Grado (TFG)](/docs/tfg.pdf).

---

## Tabla de contenidos
 
1. [Descripción y motivación](#descripción-y-motivación)
2. [Estructura del repositorio](#estructura-del-repositorio)
3. [Datasets](#datasets)
4. [Configuración del entorno](#configuración-del-entorno)
5. [Sistema de configuración (Hydra)](#sistema-de-configuración-hydra)
6. [MLflow y MinIO](#mlflow-y-minio)
7. [Pipeline paso a paso](#pipeline-paso-a-paso)
8. [Protocolos de evaluación: T1 y T2](#protocolos-de-evaluación-t1-y-t2)
9. [Resultados](#resultados)
10. [Licencia y citación](#licencia-y-citación)
---
 
## Descripción y motivación
 
Los sistemas de detección de pólipos basados en *deep learning* suelen evaluarse mediante validación cruzada o partición *train/test* dentro del **mismo** dataset. Esa metodología sobreestima sistemáticamente el rendimiento real, porque no captura cómo se comporta el modelo ante datos procedentes de otro centro, otro endoscopio o otra población de pacientes, el escenario habitual en un despliegue clínico real.
 
Este proyecto evalúa esa brecha de generalización de forma sistemática:
 
- Se entrena un conjunto homogéneo de modelos YOLO sobre un único dataset de referencia.
- Se evalúa ese mismo conjunto de modelos, **sin reentrenar**, sobre dos datasets externos que representan distintos grados de cambio de distribución (protocolos **T1** y **T2**, ver más abajo).
- Se repite la evaluación variando la resolución de entrada y la precisión numérica (FP32, FP16, INT8), tanto en **servidor** (GPU de escritorio/servidor) como en un dispositivo **edge real (NVIDIA Jetson)**.
El objetivo principal es determinar qué combinación de modelo + resolución + precisión ofrece el mejor compromiso entre **precisión diagnóstica** y **rendimiento en tiempo de inferencia** en un dispositivo embebido, condición necesaria para un sistema de apoyo a la colonoscopia en tiempo real.
Todos los experimentos se trazan con **MLflow** (métricas, parámetros, artefactos) y los artefactos pesados (modelos, datasets exportados) se almacenan en un backend **MinIO (S3-compatible)**, para que el flujo de trabajo sea reproducible y auditable de extremo a extremo.
 
---
 
## Estructura del repositorio
 
```
polyp-detection-meta-analysis/
├── src/
│   ├── configs/                  # Configuraciones Hydra (YAML), ver sección dedicada
│   │   ├── conf.yaml              # Config raíz: compone los grupos siguientes + hiperparámetros comunes (bloque params)
│   │   ├── files/
│   │   │   └── polyp.yaml          # Rutas de datasets crudos y definición de los protocolos T1/T2
│   │   ├── connection/
│   │   │   └── jetson.yaml         # Credenciales SSH/host de la Jetson para evaluación remota
│   │   ├── mlflow/
│   │   │   └── mlflow.yaml         # URIs y credenciales de MLflow + MinIO
│   │   ├── experiment/             # Un YAML por experimento: modelo y pesos iniciales
│   │   └── hpo/                    # Configuración de búsqueda de hiperparámetros (Optuna sweeper)
|   |── data/                       # Carpeta de utilidades para procesar los datasets crudos
│   ├── utils/                      # Funciones auxiliares para usar con la Jetson
│   ├── preprocess.py               # Construye los splits train/val/test de cada protocolo a partir de los datasets crudos
│   ├── train.py                    # Entrenamiento (entry point de Hydra, soporta -m / multirun)
│   ├── test.py                     # Evaluación en servidor sobre los protocolos T1 / T2
│   ├── trigger_jetson_test.py     # Lanza remotamente la evaluación en la Jetson (vía fabric/SSH)
│   └── jetson_test.py             # Entry point que se ejecuta DENTRO del contenedor de la Jetson
├── ultralytics/                   # Fork propio de Ultralytics (dependencia editable)
├── Dockerfile                     # Imagen para la Jetson (basada en l4t-ml)
├── requirements.txt               # Dependencias del entorno de servidor/desarrollo
├── full_pipeline.sh                # Orquesta preprocesado → entrenamiento → tests local/Jetson
├── launch_experiments.sh          # Lanza todos los entrenamientos definidos en configs/experiments
├── launch_local_tests.sh          # Lanza la evaluación T1/T2 en servidor (multirun)
├── launch_jetson_tests.sh         # Lanza la evaluación T1/T2 en la Jetson (multirun)
└── README.md
```
 
> **Nota:** la carpeta `src/configs` es el corazón del proyecto. `conf.yaml` compone, mediante grupos de Hydra, la configuración de datasets/protocolos (`files/polyp.yaml`), conexión a la Jetson (`connection/jetson.yaml`) y MLflow/MinIO (`mlflow/mlflow.yaml`), además de fijar en su bloque `params` los hiperparámetros comunes a todos los entrenamientos. Cada YAML dentro de `experiment/` solo define qué modelo entrenar y desde qué pesos parte. Ver la sección [Sistema de configuración (Hydra)](#sistema-de-configuración-hydra) para el detalle de cada grupo.
 
---
 
## Datasets
 
El estudio usa **tres** conjuntos de datos de colonoscopia, todos reconvertidos a formato de detección de objetos (cajas delimitadoras alrededor del pólipo) a partir de sus máscaras/anotaciones originales. Las rutas y la definición de cada dataset y protocolo viven en `src/configs/files/polyp.yaml`. Para descargarlos hace falta solicitar acceso a sus autores/repositorios oficiales; este repo **no redistribuye** ninguno de ellos.
 
| Dataset crudo | Tipo de anotación original | Rol en el estudio |
|---|---|---|
| **CVC-ClinicDB** | Máscaras binarias (`type: mask`) | Train + validación (in-distribution) en **ambos** protocolos |
| **CVC-ColonDB** | Máscaras binarias (`type: mask`) | Test **OOD** del protocolo **T1** |
| **SUN Colonoscopy Video Database** | Anotaciones por caso, en `.txt` (`type: sun_annotation`), con frames positivos y negativos | Test **OOD** del protocolo **T2** |
 
> El dataset de **entrenamiento es siempre CVC-ClinicDB** en los dos protocolos: lo que cambia entre T1 y T2 es exclusivamente el conjunto sobre el que se evalúa fuera de distribución (*out-of-distribution*, OOD), no el conjunto de entrenamiento. Ver la sección [Protocolos de evaluación: T1 y T2](#protocolos-de-evaluación-t1-y-t2) para el detalle de cada *split*.
 
### Dónde obtener los datasets
 
- **CVC-ClinicDB**: disponible a través del [CVC (Computer Vision Center)](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/) o de réplicas en repositorios académicos de detección de pólipos (p. ej. Kaggle).
- **CVC-ColonDB**: igualmente disponible a través del [CVC](http://www.cvc.uab.es/CVC-Colon/index.php/databases/).
- **SUN Colonoscopy Video Database**: requiere solicitud formal a través del formulario oficial del [SUN database (Showa University & Nagoya University)](http://amed8k.sundatabase.org/), ya que es un dataset con acceso restringido por convenio.
### Estructura esperada en disco
 
`src/configs/files/polyp.yaml` define, bajo `base_path`, dónde se espera encontrar cada dataset crudo y dónde se cachean los resultados procesados por `src/preprocess.py`:
 
```yaml
base_path: /ruta/a/tus/datasets
raw_datasets:
  clinic:                                       # CVC-ClinicDB
    type: mask
    images_dir: CVC-ClinicDB/PNG/Original
    mask_dir: CVC-ClinicDB/PNG/GroundTruth
    metadata_file: CVC-ClinicDB/metadata.csv
    duplicate_threshold: 0.04
    cache_dir: processed_datasets/clinic
  colon:                                        # CVC-ColonDB
    type: mask
    images_dir: CVC-ColonDB/CVC-ColonDB/images
    mask_dir: CVC-ColonDB/CVC-ColonDB/masks
    duplicate_threshold: 0.04
    cache_dir: processed_datasets/colon
  sun:                                           # SUN
    type: sun_annotation
    positive_dir: sun/sundatabase_positive
    negative_dir: sun/sundatabase_negative
    annotation_dir: sun/sundatabase_positive/annotation_txt
    metadata_file: sun/metadata.csv
    duplicate_threshold: 0.07
    cache_dir: processed_datasets/sun
```
 
Es decir, dentro de `base_path` debes colocar las carpetas `CVC-ClinicDB/`, `CVC-ColonDB/` y `sun/` respetando la estructura interna que cada dataset publica oficialmente (imágenes en PNG + máscaras de ground truth para los CVC; carpetas de frames positivos/negativos + anotaciones `.txt` para SUN). `src/preprocess.py` lee esas rutas, deduplica imágenes casi idénticas (`duplicate_threshold`, por similitud) y genera los `cache_dir` correspondientes ya en formato YOLO. Ajusta `base_path` a tu copia local antes de ejecutar nada.
 
---
 
## Configuración del entorno
 
Hay dos entornos distintos en este proyecto: el de **servidor** (entrenamiento + evaluación local, normalmente con GPU de escritorio/servidor) y el de **Jetson** (evaluación embebida), construido como imagen Docker.
 
### Opción A — Entorno de servidor (virtualenv)
 
Requisitos: Python ≥ 3.10, GPU NVIDIA con drivers y CUDA instalados (recomendado para entrenamiento), `git`.
 
```bash
git clone https://github.com/Sergiooo0/polyp-detection-meta-analysis.git
cd polyp-detection-meta-analysis
 
python3 -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
 
pip install --upgrade pip
pip install -r requirements.txt
```
 
`requirements.txt` instala, entre otros:
 
- `hydra-core` — gestión de configuración y *multirun*.
- `hydra-optuna-sweeper` + `optuna` — búsqueda de hiperparámetros integrada con Hydra.
- `mlflow` — tracking de experimentos.
- `boto3` — cliente S3, usado por MLflow para hablar con MinIO.
- `ultralytics` — **fork propio** ([Sergiooo0/ultralytics](https://github.com/Sergiooo0/ultralytics)), instalado directamente desde GitHub. Si vas a modificarlo, clónalo aparte y reinstálalo en modo editable: `pip install -e ./ultralytics`.
- `fabric` — orquestación remota por SSH, usada para disparar la evaluación en la Jetson desde el servidor.
### Opción B — Entorno Jetson (Docker)
 
La evaluación en el dispositivo embebido se ejecuta dentro de un contenedor construido sobre la imagen oficial de NVIDIA para Jetson Linux (L4T) con ML preinstalado (`nvcr.io/nvidia/l4t-ml`), por lo que **debe construirse y ejecutarse en la propia Jetson** (o en un entorno con emulación ARM64 + runtime NVIDIA equivalente).
 
Requisitos en la Jetson:
 
- JetPack compatible con `l4t-ml:r36.2.0-py3`.
- `nvidia-docker` / NVIDIA Container Runtime habilitado.
Construcción de la imagen (desde la raíz del repo, con la carpeta `ultralytics/` presente):
 
```bash
docker build -t polyp-jetson-test .
```
 
El `Dockerfile`:
 
- Instala dependencias del sistema (`libgl1-mesa-glx`, `python3-pip`).
- Instala el fork de `ultralytics` en modo editable.
- Instala `mlflow`, `boto3`, `jetson-stats` (telemetría de la Jetson: uso de GPU, temperatura, consumo), `fabric` (orquestación remota por SSH) y `onnxslim` + `onnxruntime` (exportación/inferencia ONNX, necesaria para los backends de precisión reducida).
- Define como `ENTRYPOINT` el script `src/jetson_test.py`.
Ejecución típica (ver también [Pipeline paso a paso](#pipeline-paso-a-paso)):
 
```bash
docker run --runtime nvidia --rm \
  -e AWS_ACCESS_KEY_ID=<usuario_minio> \
  -e AWS_SECRET_ACCESS_KEY=<password_minio> \
  -v /ruta/a/datasets:/app/data \
  polyp-jetson-test \
  test.img_size=416 test.precision_mode=FP32 test.protocol=t1
```
 
> En la práctica, no se invoca `docker run` a mano: es `src/trigger_jetson_test.py` (ejecutado desde el servidor) el que se conecta a la Jetson por SSH usando **`fabric`** — con las credenciales definidas en `src/configs/connection/jetson.yaml` — y dispara remotamente el contenedor para cada combinación de parámetros del *multirun*. Ver `launch_jetson_tests.sh`.
 
### `src/configs/connection/jetson.yaml`
 
Este archivo define cómo llegar a la Jetson desde el servidor:
 
```yaml
host: IP_ADDRESS
port: PORT
username: USERNAME
password: PASSWORD
test_folder_remote: /path/to/remote/test_folder
```
 
- `host` / `port` / `username` / `password`: credenciales SSH de la Jetson. `trigger_jetson_test.py` los usa con `fabric` para conectarse, copiar lo necesario y lanzar el contenedor Docker remotamente.
- `test_folder_remote`: ruta **en la propia Jetson** donde debe estar disponible el dataset de test (el protocolo correspondiente) para que el contenedor lo monte/lea durante la evaluación.
Sustituye los cuatro primeros campos por los datos reales de tu dispositivo, y asegúrate de que `test_folder_remote` existe en la Jetson y contiene los datos del protocolo que vayas a evaluar. Por seguridad, considera mover `password` a una variable de entorno interpolada con `${oc.env:...}` (como ya se hace en `mlflow/mlflow.yaml`) en lugar de dejarla en texto plano si vas a subir este archivo a un repositorio compartido.
 
---
 
## Sistema de configuración (Hydra)
 
Todo el proyecto — entrenamiento y evaluación — se controla mediante **[Hydra](https://hydra.cc/)**. La configuración está organizada en **grupos**, cada uno en su propia subcarpeta de `src/configs/`, que `conf.yaml` compone como configuración raíz:
 
```
src/configs/
├── conf.yaml          # Configuración raíz: compone los grupos siguientes + hiperparámetros comunes
├── files/
│   └── polyp.yaml      # Datasets crudos y definición de los protocolos T1/T2 (ver sección Datasets)
├── connection/
│   └── jetson.yaml     # Host/credenciales SSH de la Jetson para evaluación remota
├── mlflow/
│   └── mlflow.yaml      # URIs y credenciales de MLflow + MinIO
├── experiment/         # Un YAML por experimento: modelo a entrenar y pesos de partida
└── hpo/                 # Configuración de búsqueda de hiperparámetros (hydra-optuna-sweeper)
```
 
### `conf.yaml`: configuración raíz
 
```yaml
defaults:
  - _self_
  - files: polyp
  - connection: jetson
  - mlflow: mlflow
  - experiment: null
 
params:
  protocol: t1
  seed: 42
  val_ratio: 0.2
  model: yolo11n.yaml
  pretrained_weights: null
  model_name: yolom
  img_size: 640
  epochs: 250
  batch_size: 32
  lr: 0.0015
  optimizer: AdamW
  device: [0, 1]
  # ... resto de hiperparámetros de entrenamiento y data augmentation
 
test:
  protocol: t1
  img_size: 640
  precision_mode: FP16
  metric: val_AP50_95
  top_k: 50
  # ...
```
 
El bloque `defaults` indica qué archivo concreto de cada grupo se carga por defecto (`files: polyp` → `files/polyp.yaml`, `connection: jetson` → `connection/jetson.yaml`, `mlflow: mlflow` → `mlflow/mlflow.yaml`); `experiment: null` significa que, si no se especifica `experiment=<algo>` por línea de comandos, no se sobreescribe ningún parámetro de modelo y se usan los valores por defecto de `params` (en este caso, `yolo11n.yaml` sin pesos preentrenados).
 
El bloque `params` reúne **todos los hiperparámetros de entrenamiento** (optimizador, *learning rate*, *batch size*, número de épocas, *data augmentation*, etc.) y también el **protocolo activo** (`params.protocol`) y la **semilla** (`params.seed`). La idea del estudio es mantener estos hiperparámetros fijos entre modelos, variando entre ejecuciones principalmente `params.seed` (para repetir cada entrenamiento con varias semillas y poder reportar resultados con variabilidad, media ± desviación) y, cuando corresponde, `params.protocol`.
 
El bloque `test` reúne los parámetros de evaluación (resolución, modo de precisión, protocolo, métrica de selección de mejores *checkpoints*, etc.), usados por `src/test.py` y `src/jetson_test.py`.
 
### Grupo `experiment/`: qué modelo entrenar
 
Cada fichero en `src/configs/experiment/` sobreescribe únicamente `params.model` (la arquitectura, p. ej. `yolo11n.yaml`, `yolo11s.yaml`, `yolov5n.yaml`, `yolov8n.yaml`…) y `params.pretrained_weights` (si se parte de pesos preentrenados en COCO o de inicialización propia). Algunos de los YAML disponibles actualmente:
 
```
yolo11_n.yaml           yolo11_n_polyp.yaml       yolo11_n_polypTST.yaml   yolo11_n_tst.yaml
yolo11_s.yaml           yolo11_s_polyp.yaml       yolo11_s_polypTST.yaml   yolo11_s_tst.yaml
yolo5_n.yaml            yolo5_n_polyp.yaml        yolo5_n_tst.yaml
yolo5_s.yaml            yolo5_s_polyp.yaml        yolo5_s_tst.yaml
yolo8_nano.yaml         yolo8_small.yaml
```
 
> No defino aquí el contenido exacto de cada variante (`_polyp`, `_polypTST`, `_tst`, etc.) porque corresponde a tu convención interna de nombres — documenta brevemente en esta sección, o en comentarios dentro de cada YAML, qué distingue a cada sufijo (p. ej. si `_polyp` parte de pesos ya afinados en datos médicos, o si `_tst` es una variante reducida para pruebas rápidas).
 
`launch_experiments.sh` recoge automáticamente **todos** los YAML de `experiment/` y los lanza como un *multirun* de Hydra, cruzándolos con las semillas:
 
```bash
#!/bin/bash
SEEDS="42,55,66"
FOLDER="src/configs/experiment"
files=()
for file in "$FOLDER"/*.yaml; do
  files+=("$(basename "$file")")
done
EXPERIMENTS=$(IFS=,; echo "${files[*]}")
 
python src/train.py -m \
  experiment=$EXPERIMENTS \
  params.seed=$SEEDS \
  params.experiment_name=polyp_detection
```
 
Esto entrena **cada experimento (modelo + pesos de partida) × cada seed**, siempre sobre el dataset y protocolo activos en `params` y con los mismos hiperparámetros de `conf.yaml`. Para lanzar solo un subconjunto, descomenta y edita la línea `EXPERIMENTS="..."` dentro de `launch_experiments.sh`, o invoca `train.py` directamente:
 
```bash
python src/train.py experiment=yolo11_n_tst.yaml params.seed=42 params.experiment_name=polyp_detection
```
Ejecutando `python train.py` sin argumentos, Hydra carga `conf.yaml` y entrena el modelo por defecto (`yolo11n.yaml`) con los hiperparámetros de `params`.

### Grupo `hpo/`: búsqueda de hiperparámetros con Optuna
 
Cuando se quiere explorar el espacio de hiperparámetros en lugar de fijarlo (p. ej. para encontrar el mejor *learning rate* o *batch size* antes de congelar la configuración de `conf.yaml` usada en el estudio principal), `src/configs/hpo/` contiene la configuración del *sweeper* de Optuna integrado vía `hydra-optuna-sweeper`. Se invoca añadiendo `hydra/sweeper=optuna` (o el override equivalente que definas en el YAML de `hpo/`) a la llamada a `train.py -m`.
 
### Grupos `connection/` y `mlflow/`: ver sección dedicada
 
La configuración de conexión a la Jetson (`connection/jetson.yaml`) y de MLflow/MinIO (`mlflow/mlflow.yaml`) se explica en detalle en la siguiente sección, [MLflow y MinIO](#mlflow-y-minio), ya que ahí también se necesitan credenciales sensibles.
 
### Overrides desde línea de comandos
 
Como con cualquier configuración Hydra, cualquier campo puede sobreescribirse desde línea de comandos (`clave=valor`) sin tocar los YAML — por ejemplo, para un hiperparámetro puntual sin editar `conf.yaml`:
 
```bash
python src/train.py params.epochs=150
```
 
Hydra guarda automáticamente, por cada *run*, la configuración resuelta y los logs en su propio directorio de salida (`outputs/` o `multirun/` por defecto).
 
### Configuración de evaluación (protocolos)
 
La evaluación usa el mismo mecanismo de *multirun* sobre `src/test.py`, barriendo:
 
- `test.img_size` — resolución de entrada (p. ej. `640,416,320`).
- `test.precision_mode` — `FP32`, `FP16` y, en Jetson, también `INT8`, `ONNX-FP32`, `ONNX-FP16` y `ONNX-INT8`
- `test.protocol` — `t1`, `t2` (ver [siguiente sección](#protocolos-de-evaluación-t1-y-t2)).
```bash
python src/test.py -m \
  test.img_size=640,416,320 \
  test.precision_mode=FP32,FP16 \
  test.protocol=t1,t2
```
 
Como con cualquier configuración Hydra, cualquier campo puede sobreescribirse desde línea de comandos (`clave=valor`) sin tocar los YAML, y Hydra guarda automáticamente, por cada *run*, la configuración resuelta y los logs en su propio directorio de salida (`outputs/` o `multirun/` por defecto).
 
---
 
## MLflow y MinIO
 
Todos los experimentos (entrenamiento y evaluación) registran parámetros, métricas y artefactos en **MLflow**. Los artefactos (pesos `.pt`/`.onnx`, gráficas, predicciones) se almacenan en un bucket **MinIO**, que actúa como backend de almacenamiento S3-compatible para MLflow. Toda la configuración de conexión vive en `src/configs/mlflow/mlflow.yaml`.
 
### 1. Levantar MLflow + MinIO
 
Si todavía no tienes un servidor de MLflow/MinIO corriendo, la forma más simple es con Docker Compose en la máquina que actuará como servidor. Ejemplo orientativo (ajusta credenciales y puertos a tu entorno; este `docker-compose.yml` no forma parte del repo, es solo una referencia para levantar el servicio):
 
```yaml
version: "3.8"
services:
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"   # API S3
      - "9001:9001"   # Consola web
    environment:
      MINIO_ROOT_USER: <usuario_minio>
      MINIO_ROOT_PASSWORD: <password_minio>
    volumes:
      - minio_data:/data
 
  mlflow:
    image: ghcr.io/mlflow/mlflow
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:///mlflow.db
      --default-artifact-root s3://mlflow-artifacts/
    ports:
      - "5000:5000"
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: <usuario_minio>
      AWS_SECRET_ACCESS_KEY: <password_minio>
    depends_on:
      - minio
 
volumes:
  minio_data:
```
 
```bash
docker compose up -d
```
 
Después, crea el bucket `mlflow-artifacts` desde la consola de MinIO (`http://<host>:9001`) o con el cliente `mc`:
 
```bash
mc alias set local http://<host>:9000 <usuario_minio> <password_minio>
mc mb local/mlflow-artifacts
```
 
### 2. `src/configs/mlflow/mlflow.yaml`: configuración del cliente
 
Este archivo es el que usan tanto `train.py`/`test.py` (servidor) como `jetson_test.py` (Jetson) para saber a qué MLflow y MinIO conectarse:
 
```yaml
# Local tracking DB path. If null, train.py will use repository mlruns.db
tracking_uri: http://[IP_ADDRESS]:5000
# MinIO server URL for MLflow artifact storage
s3_endpoint_url: http://[IP_ADDRESS]:9000
aws_access_key_id: ${oc.env:AWS_ACCESS_KEY_ID,username}
aws_secret_access_key: ${oc.env:AWS_SECRET_ACCESS_KEY,password}
artifact_location: s3://mlflow-artifacts/
```
 
Antes de lanzar nada:
 
1. Sustituye `[IP_ADDRESS]` en `tracking_uri` y `s3_endpoint_url` por la IP (o nombre de host) real donde corren tu MLflow y tu MinIO.
2. Las credenciales (`aws_access_key_id` / `aws_secret_access_key`) usan la sintaxis de interpolación de Hydra/OmegaConf `${oc.env:VARIABLE,valor_por_defecto}`: si las variables de entorno `AWS_ACCESS_KEY_ID` y `AWS_SECRET_ACCESS_KEY` están definidas en el shell donde lanzas `train.py`/`test.py` (o en el `docker run` de la Jetson), se usan esas; si no, caen al valor por defecto indicado (`username`/`password` en el ejemplo del repo — **cámbialos** o, mejor, exporta siempre las variables de entorno en vez de dejar el valor por defecto en texto plano):
```bash
export AWS_ACCESS_KEY_ID=<usuario_minio>
export AWS_SECRET_ACCESS_KEY=<password_minio>
```
 
3. Asegúrate de que el bucket indicado en `artifact_location` (`mlflow-artifacts` por defecto) existe en tu MinIO.
> Estas mismas variables de entorno deben pasarse también al contenedor de la Jetson (`docker run -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=...`), ya que el dispositivo se conecta por red al MLflow/MinIO que corre en el servidor.
 
### 3. Verificación
 
- UI de MLflow: `http://<IP_ADDRESS>:5000` — deberías ver un experimento llamado `polyp_detection` (o el valor de `params.experiment_name`) con un *run* por cada combinación de modelo/seed/protocolo.
- Consola de MinIO: `http://<IP_ADDRESS>:9001` — deberías ver el bucket `mlflow-artifacts` poblándose con los artefactos de cada *run* (pesos, curvas, métricas exportadas).
> Si las credenciales no son correctas o `s3_endpoint_url` no apunta a tu MinIO, MLflow puede fallar al subir artefactos o intentar usar el S3 de AWS real — comprueba siempre `mlflow/mlflow.yaml` (y las variables de entorno que interpola) antes de lanzar un experimento largo.
 
---
 
## Pipeline paso a paso
 
El flujo completo (también resumido en `full_pipeline.sh`) es:
 
```
preprocesamiento → entrenamiento → evaluación en servidor (T1/T2) → evaluación en Jetson (T1/T2)
```
 
### 0. Requisitos previos
 
- Entorno de servidor configurado (ver [Configuración del entorno](#configuración-del-entorno)).
- MLflow + MinIO levantados y accesibles (ver [MLflow y MinIO](#mlflow-y-minio)).
- Datasets descargados y organizados según [Datasets](#datasets).
- Si vas a evaluar en Jetson: imagen Docker construida en el dispositivo y accesible por SSH desde el servidor (`trigger_jetson_test.py` usa `fabric`, por lo que necesitarás credenciales SSH configuradas, p. ej. en `~/.ssh/config` o pasadas como parámetros de Hydra).
### 1. Preprocesamiento
 
```bash
python src/preprocess.py
```
 
Convierte/organiza los datasets brutos al formato esperado por Ultralytics (estructura `images/`+`labels/` y `data.yaml`), y aplica cualquier limpieza, división train/val/test o normalización necesaria. Revisa `src/configs/` para las rutas de entrada/salida que usa este script en tu copia local.
 
### 2. Entrenamiento
 
```bash
./launch_experiments.sh
```
 
Esto lanza, vía Hydra *multirun*, **todos** los experimentos definidos en `src/configs/experiments/` para cada *seed* en `SEEDS="42,55,66"`. Cada combinación experimento×seed es un *run* independiente, registrado en MLflow bajo el experimento `polyp_detection`, con sus pesos finales subidos a MinIO como artefacto.
 
Para un único experimento puntual:
 
```bash
python src/train.py experiment=<nombre_yaml> params.seed=42 params.experiment_name=polyp_detection
```
 
### 3. Evaluación en servidor (T1 / T2)
 
```bash
./launch_local_tests.sh
```
 
Equivalente a:
 
```bash
python src/test.py -m \
  test.img_size=640,416,320 \
  test.precision_mode=FP32,FP16 \
  test.protocol=t1,t2
```
 
Cada modelo entrenado en el paso anterior se evalúa, **sin reentrenar**, sobre los datasets de los protocolos T1 y T2, en las tres resoluciones (640/416/320) y dos precisiones (FP32/FP16), registrando todas las métricas resultantes en MLflow.
 
### 4. Evaluación en Jetson (T1 / T2)
 
```bash
./launch_jetson_tests.sh
```
 
Equivalente a:
 
```bash
python src/trigger_jetson_test.py -m \
  test.img_size=416 \
  test.precision_mode=FP32,FP16,INT8 \
  test.protocol=t1,t2
```
 
Este paso:
 
1. Se ejecuta **desde el servidor**.
2. Por cada combinación de precisión/protocolo, se conecta por SSH (vía `fabric`) a la Jetson.
3. Lanza el contenedor Docker (`ENTRYPOINT src/jetson_test.py`) en el dispositivo, que carga el modelo correspondiente (exportado a ONNX cuando aplica, según la precisión solicitada), ejecuta la inferencia sobre el dataset del protocolo indicado y mide métricas de **rendimiento embebido** además de las de detección: latencia, FPS, uso de GPU/CPU y consumo (vía `jetson-stats`).
4. Sube los resultados a MLflow/MinIO igual que en el paso 3, normalmente bajo un experimento o etiqueta distinta (p. ej. `device=jetson`) para poder comparar servidor vs. edge.
### Ejecutar todo de una vez
 
```bash
./full_pipeline.sh
```
 
equivalente a:
 
```bash
python preprocess.py
./launch_experiments.sh
./launch_local_tests.sh
./launch_jetson_tests.sh
```
 
> Ten en cuenta que un pipeline completo puede tardar horas/días según el número de experimentos, *seeds*, resoluciones y precisiones — está pensado para lanzarse en background (`nohup`, `tmux`, `screen`, etc.) en una máquina dedicada.
 
---
 
## Protocolos de evaluación: T1 y T2
 
La pregunta central del estudio es: **¿cómo se degrada un modelo entrenado en un dataset cuando se evalúa en datos que no ha visto nunca, procedentes de otra fuente?** Para responderla de forma graduada, se definen dos protocolos de test que comparten exactamente el mismo conjunto de modelos entrenados, pero cambian el dataset de evaluación:
 
| Protocolo | Dataset de test | Magnitud del *domain shift* | Qué mide |
|---|---|---|---|
| **T1** | **CVC-ColonDB / CVC-ClinicDB** | **Moderado** | Generalización ante un cambio de fuente "razonable": mismo tipo de adquisición (colonoscopia óptica estándar), pero distinto centro/equipo/cohorte de pacientes que el dataset de entrenamiento. |
| **T2** | **SUN Colonoscopy Video Database** | **Severo** | Generalización ante un cambio de fuente mucho más exigente: distinta procedencia de los vídeos, mayor diversidad de condiciones de captura (iluminación, calidad, tipos de pólipo, prevalencia de fotogramas sin pólipo), lo que se traduce en una distribución de datos notablemente distinta a la de entrenamiento. |
 
Importante: **en ningún caso se reentrena ni se ajusta (fine-tuning) el modelo entre protocolos**. Tanto T1 como T2 parten del mismo checkpoint entrenado únicamente sobre el dataset de entrenamiento; lo único que cambia es el dataset sobre el que se mide. Esto permite aislar el efecto del *domain shift* del efecto de cualquier adaptación del modelo, y comparar directamente:
 
- **T1 vs. entrenamiento** → coste de generalizar ante un shift moderado.
- **T2 vs. T1** → coste adicional de un shift severo, y si ese coste es proporcional o se agrava de forma no lineal según arquitectura/tamaño del modelo.
Ambos protocolos se ejecutan, además, cruzados con:
 
- **Resolución de entrada** (`test.img_size`): 640 / 416 / 320 en servidor; 416 en Jetson (resolución fijada en el dispositivo embebido por restricciones de cómputo/latencia, ajustable en `launch_jetson_tests.sh`).
- **Precisión numérica** (`test.precision_mode`): FP32 y FP16 en servidor; FP32, FP16 e INT8 en Jetson — la cuantización INT8 solo se evalúa en el dispositivo embebido porque es ahí donde aporta el mayor beneficio práctico (latencia/consumo) y donde es relevante medir su impacto en la métrica de detección.
El cruce protocolo × resolución × precisión × *seed* × modelo es el diseño experimental completo del estudio, y es lo que generan conjuntamente `launch_local_tests.sh` y `launch_jetson_tests.sh`.
 
---

## Resultados

Los resultados finales están disponibles para consultar en la sección de **Resultados y Discusión** del [TFG](docs/tfg.pdf).

---

## Licencia

Este proyecto está licenciado bajo GNU GPL v3.0. Consulta el archivo LICENSE para más detalles.

Copyright (c) 2026 Sergiooo0 .
