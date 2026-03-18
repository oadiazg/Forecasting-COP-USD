# Forecasting COP/USD con DFGCN

## Descripción del Proyecto

Este repositorio implementa el modelo **DFGCN (Dual Frequency Graph Convolutional Network)** para la predicción de la tasa de cambio **Peso Colombiano / Dólar Estadounidense (COP/USD)**.

DFGCN es un modelo de aprendizaje profundo diseñado para predicción multivariada de series de tiempo. Combina dos componentes de Redes Neuronales de Grafos (GCN) que operan en frecuencias duales:

1. **GNN_time**: Modela las dependencias temporales entre parches (patches) de la serie de tiempo, construyendo un grafo dinámico basado en correlación de Pearson entre segmentos temporales.
2. **GNN_variate**: Modela las dependencias entre variables (canales), también mediante correlación de Pearson, capturando relaciones entre distintos indicadores financieros.

Ambas ramas se fusionan al final para producir una predicción robusta que captura tanto la estructura temporal como las interdependencias entre variables.

---

## ¿Por qué Python?

Python es la herramienta estándar para el desarrollo de modelos de aprendizaje profundo y análisis de datos por varias razones:

- **PyTorch**: La librería de deep learning más utilizada en investigación, con soporte para GPU, diferenciación automática y módulos reutilizables (`nn.Module`).
- **PyTorch Geometric**: Extiende PyTorch con soporte nativo para Redes Neuronales de Grafos (GNN/GCN), lo que permite construir y entrenar grafos dinámicos de forma eficiente.
- **Ecosistema científico**: `NumPy`, `Pandas`, `Scikit-learn` y `Matplotlib` forman un ecosistema maduro para manipulación de datos, preprocesamiento, evaluación y visualización.
- **Facilidad de prototipado**: Python permite iterar rápidamente sobre ideas y arquitecturas de modelos con código limpio y legible.
- **Comunidad y recursos**: La mayoría de los papers de investigación en series de tiempo financieras publican su código en Python.

---

## Estructura del Repositorio

```
Forecasting-COP-USD/
│
├── modelos/                     # Arquitecturas de modelos
│   ├── DFGCN.py                 # Modelo principal DFGCN
│   ├── RevIN.py                 # Normalización Reversible de Instancia
│   └── RandomWalk.py            # Modelo de Caminata Aleatoria (benchmark)
│
├── layers/                      # Capas de la red neuronal
│   ├── Embed.py                 # Capas de embedding (posicional, temporal, token)
│   ├── GNN_time.py              # GCN para dependencias temporales
│   ├── GNN_variate.py           # GCN para dependencias entre variables
│   └── Transformer_encoder.py  # Encoder Transformer con atención global
│
├── experiments/                 # Pipelines de entrenamiento y evaluación
│   ├── exp_basic.py             # Clase base de experimentos
│   ├── exp_term_forecasting.py  # Pipeline principal (train/vali/test)
│   └── exp_long_term_forecasting_partial.py  # Entrenamiento parcial (zero-shot)
│
├── data_provider/               # Carga y preparación de datos
│   ├── data_factory.py          # Selector de dataset según argumentos
│   └── data_loader.py           # Clases Dataset para distintos formatos
│
├── utils/                       # Utilidades
│   ├── metrics.py               # Métricas de evaluación (MAE, MSE, RMSE, MAPE, RSE)
│   ├── tools.py                 # Herramientas: EarlyStopping, ajuste de LR, visualización
│   ├── timefeatures.py          # Características temporales (hora, día, mes, etc.)
│   └── masking.py               # Máscaras causales para atención
│
├── datos/                       # 📁 Carpeta para sus datos (ponga aquí sus archivos CSV)
│
├── run.py                       # Script principal de ejecución
├── requirements.txt             # Dependencias de Python
└── README.md                    # Este archivo
```

---

## Formato de los Datos

### Estructura del archivo CSV

Coloque sus datos en la carpeta `datos/`. El archivo debe seguir este formato:

```csv
date,variable1,variable2,...,variableN,COP_USD
2020-01-01,valor1,valor2,...,valorN,3750.50
2020-01-02,valor1,valor2,...,valorN,3762.30
...
```

**Requisitos obligatorios:**
| Campo | Descripción |
|-------|-------------|
| `date` | Columna de fecha (primera columna). Formatos aceptados: `YYYY-MM-DD`, `YYYY-MM-DD HH:MM:SS` |
| Columnas numéricas | Las demás columnas deben ser numéricas (float o int). No se admiten valores nulos (se recomienda interpolación previa) |
| Variable objetivo | Debe ser la **última columna** del CSV, o indicarla con el parámetro `--target` |

**Ejemplo para tasa COP/USD diaria:**
```csv
date,PIB_col,tasa_interes,inflacion,petroleo_brent,COP_USD
2020-01-02,0.023,4.25,3.80,68.50,3730.45
2020-01-03,0.023,4.25,3.80,67.80,3745.20
2020-01-06,0.023,4.25,3.82,68.10,3758.90
```

**Ejemplo univariado (solo la tasa):**
```csv
date,COP_USD
2020-01-02,3730.45
2020-01-03,3745.20
2020-01-06,3758.90
```

### División de datos

El modelo divide automáticamente los datos en:
- **Entrenamiento**: 70% del total
- **Validación**: 10% del total
- **Prueba/Test**: 20% del total

> ⚠️ Se recomienda un mínimo de **500 observaciones** para un entrenamiento adecuado. Para datos diarios, esto equivale a ~2 años de historia.

### Frecuencias disponibles (`--freq`)
| Código | Descripción |
|--------|-------------|
| `d`    | Diaria (recomendada para COP/USD) |
| `b`    | Días hábiles |
| `h`    | Horaria |
| `t`    | Por minuto |
| `w`    | Semanal |
| `m`    | Mensual |

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/oadiazg/Forecasting-COP-USD.git
cd Forecasting-COP-USD
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota sobre PyTorch Geometric**: Si tiene problemas con `torch-geometric`, instale las dependencias manualmente siguiendo las instrucciones en https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

---

## Uso del Modelo DFGCN

### Modo 1: Entrenamiento

Para entrenar el modelo con sus datos propios:

```bash
python run.py \
  --is_training 1 \
  --model_id COP_USD_experimento1 \
  --model DFGCN \
  --data custom \
  --root_path ./datos/ \
  --data_path tasa_cop_usd.csv \
  --features M \
  --target COP_USD \
  --freq d \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 30 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --n_heads 1 \
  --e_layers 1 \
  --d_ff 128 \
  --patch_len 8 \
  --k 2 \
  --use_norm 1 \
  --train_epochs 20 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --patience 5 \
  --lradj type7 \
  --dropout 0.1 \
  --des entrenamiento_inicial
```

Para series **univariadas** (solo la tasa COP/USD):

```bash
python run.py \
  --is_training 1 \
  --model_id COP_USD_uni \
  --model DFGCN \
  --data custom \
  --root_path ./datos/ \
  --data_path tasa_cop_usd.csv \
  --features S \
  --target COP_USD \
  --freq d \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 30 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 64 \
  --n_heads 1 \
  --e_layers 1 \
  --d_ff 64 \
  --patch_len 8 \
  --k 1 \
  --use_norm 1 \
  --train_epochs 20 \
  --batch_size 32 \
  --learning_rate 0.0001
```

### Modo 2: Validación

La validación se ejecuta automáticamente al final del entrenamiento. Para ejecutar **solo la validación/prueba** sobre un modelo ya entrenado:

```bash
python run.py \
  --is_training 0 \
  --model_id COP_USD_experimento1 \
  --model DFGCN \
  --data custom \
  --root_path ./datos/ \
  --data_path tasa_cop_usd.csv \
  --features M \
  --target COP_USD \
  --freq d \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 30 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --n_heads 1 \
  --e_layers 1 \
  --d_ff 128 \
  --patch_len 8 \
  --k 2 \
  --use_norm 1 \
  --des entrenamiento_inicial
```

> Los resultados se guardan en `./results/` y `./test_results/`. Las métricas se registran en `result_long_term_forecast.txt`.

### Modo 3: Simulación (Random Walk como benchmark)

El modelo `RandomWalk.py` permite generar simulaciones de Monte Carlo para comparar con DFGCN:

```bash
# Simulación básica con 30 días de predicción
python modelos/RandomWalk.py \
  --data_path datos/tasa_cop_usd.csv \
  --target_col COP_USD \
  --pred_len 30 \
  --num_simulations 1000

# Con más simulaciones y guardando el gráfico
python modelos/RandomWalk.py \
  --data_path datos/tasa_cop_usd.csv \
  --target_col COP_USD \
  --pred_len 60 \
  --num_simulations 5000 \
  --output_plot resultados/random_walk_60d.png

# Sin graficar (solo guardar CSV con predicciones)
python modelos/RandomWalk.py \
  --data_path datos/tasa_cop_usd.csv \
  --pred_len 30 \
  --no_plot
```

---

## Parámetros del Modelo DFGCN

### Parámetros de Datos
| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `--data` | str | `custom` | Tipo de dataset. Use `custom` para datos propios |
| `--root_path` | str | `./datos/` | Carpeta donde están los datos |
| `--data_path` | str | — | Nombre del archivo CSV |
| `--features` | str | `M` | `M`=multivariada, `S`=univariada, `MS`=multi→uni |
| `--target` | str | `OT` | Columna objetivo (para `S` o `MS`) |
| `--freq` | str | `d` | Frecuencia: `d`=diaria, `h`=horaria, `b`=hábil, etc. |

### Parámetros de la Ventana de Tiempo
| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `--seq_len` | int | `96` | Ventana de entrada: cuántos pasos históricos usa el modelo |
| `--label_len` | int | `48` | Longitud del token inicial del decoder |
| `--pred_len` | int | `30` | Horizonte de predicción: cuántos pasos a futuro predice |

### Parámetros de Arquitectura
| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `--enc_in` | int | `7` | Número de variables de entrada (debe coincidir con columnas del CSV) |
| `--d_model` | int | `128` | Dimensión interna del modelo (mayor = más capacidad, más lento) |
| `--n_heads` | int | `1` | Cabezas de atención multi-head |
| `--e_layers` | int | `1` | Número de capas del encoder |
| `--d_ff` | int | `128` | Dimensión de la capa feed-forward |
| `--patch_len` | int | `8` | Tamaño del parche temporal (divide la serie en segmentos de este tamaño) |
| `--k` | int | `2` | Número de vecinos k-NN para construir el grafo de correlación |
| `--use_norm` | int | `1` | Normalización RevIN: `1`=activa (recomendado), `0`=desactiva |
| `--dropout` | float | `0.1` | Dropout para regularización (0.0 a 1.0) |
| `--activation` | str | `sigmoid` | Función de activación de las GCN: `sigmoid` o `relu` |

### Parámetros de Entrenamiento
| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `--is_training` | int | `1` | `1`=entrenar, `0`=solo evaluar |
| `--train_epochs` | int | `10` | Número máximo de épocas |
| `--batch_size` | int | `32` | Tamaño del batch (reducir si hay problemas de memoria) |
| `--learning_rate` | float | `0.0001` | Tasa de aprendizaje del optimizador Adam |
| `--patience` | int | `3` | Épocas de paciencia para early stopping |
| `--lradj` | str | `type1` | Estrategia de ajuste de LR (ver tabla abajo) |
| `--use_amp` | flag | `False` | Precisión mixta (recomendado con GPU moderna) |

### Estrategias de Ajuste de Learning Rate (`--lradj`)
| Valor | Descripción |
|-------|-------------|
| `type1` | Reduce LR a la mitad cada época |
| `type2` | Reducción escalonada predefinida |
| `type3` | LR constante las primeras 3 épocas, luego decaimiento exponencial |
| `constant` | LR constante durante todo el entrenamiento |
| `cosine` | Decaimiento coseno suave |
| `type7` | Usa OneCycleLR (recomendado para DFGCN) |

### Parámetros de GPU
| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `--use_gpu` | bool | `True` | Usar GPU si está disponible |
| `--gpu` | int | `0` | Índice de la GPU (0 = primera) |
| `--use_multi_gpu` | flag | `False` | Entrenar en múltiples GPUs |
| `--devices` | str | `0,1,2,3` | IDs de GPUs para modo multi-GPU |

### Modos de Experimento (`--exp_name`)
| Valor | Descripción |
|-------|-------------|
| `None` | Entrenamiento y prueba estándar |
| `partial_train` | Entrenamiento con subconjunto de variables, prueba con todas |

---

## Métricas de Evaluación

Al finalizar el entrenamiento/prueba, se reportan las siguientes métricas:

| Métrica | Descripción |
|---------|-------------|
| **MAE** | Error Absoluto Medio — promedio de los errores absolutos |
| **MSE** | Error Cuadrático Medio — penaliza errores grandes |
| **RMSE** | Raíz del MSE — en las mismas unidades que la variable |
| **MAPE** | Error Porcentual Absoluto Medio — error relativo en % |
| **RSE** | Error Cuadrático Relativo — normalizado por la varianza de los datos reales |

---

## Uso del Repositorio en Visual Studio Code

### Configuración inicial en VS Code

1. **Abrir el repositorio**
   - Abra VS Code
   - `File > Open Folder...` → seleccione la carpeta `Forecasting-COP-USD`

2. **Instalar la extensión Python**
   - Vaya a `Extensions` (Ctrl+Shift+X)
   - Busque e instale **"Python"** de Microsoft

3. **Seleccionar el intérprete de Python**
   - Presione `Ctrl+Shift+P` → busque **"Python: Select Interpreter"**
   - Seleccione el entorno virtual `venv` que creó (aparecerá como `./venv/Scripts/python.exe` en Windows o `./venv/bin/python` en Linux/Mac)

4. **Abrir una terminal integrada**
   - `Terminal > New Terminal` o `` Ctrl+` ``
   - Active el entorno virtual si no está activo:
     ```bash
     # Windows
     venv\Scripts\activate
     # Linux/Mac
     source venv/bin/activate
     ```

5. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

6. **Colocar sus datos**
   - Copie su archivo CSV en la carpeta `datos/`
   - Asegúrese de que tenga columna `date` y las demás columnas numéricas

### Ejecutar el modelo desde VS Code

**Opción A: Desde la terminal integrada**

```bash
# Entrenamiento completo
python run.py --is_training 1 --model_id COP_USD --model DFGCN \
  --data custom --root_path ./datos/ --data_path tasa_cop_usd.csv \
  --features M --target COP_USD --freq d \
  --seq_len 96 --label_len 48 --pred_len 30 \
  --enc_in 7 --d_model 128 --n_heads 1 --e_layers 1 --d_ff 128 \
  --patch_len 8 --k 2 --use_norm 1 \
  --train_epochs 20 --batch_size 32 --learning_rate 0.0001 --patience 5

# Random Walk
python modelos/RandomWalk.py --data_path datos/tasa_cop_usd.csv \
  --target_col COP_USD --pred_len 30
```

**Opción B: Crear una configuración de lanzamiento (launch.json)**

1. Vaya a `Run > Add Configuration...`
2. Seleccione **Python > Python File**
3. Reemplace el contenido de `.vscode/launch.json` con:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "DFGCN - Entrenamiento",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/run.py",
            "args": [
                "--is_training", "1",
                "--model_id", "COP_USD",
                "--model", "DFGCN",
                "--data", "custom",
                "--root_path", "./datos/",
                "--data_path", "tasa_cop_usd.csv",
                "--features", "M",
                "--target", "COP_USD",
                "--freq", "d",
                "--seq_len", "96",
                "--label_len", "48",
                "--pred_len", "30",
                "--enc_in", "7",
                "--d_model", "128",
                "--n_heads", "1",
                "--e_layers", "1",
                "--d_ff", "128",
                "--patch_len", "8",
                "--k", "2",
                "--use_norm", "1",
                "--train_epochs", "20",
                "--batch_size", "32",
                "--learning_rate", "0.0001",
                "--patience", "5",
                "--lradj", "type7",
                "--dropout", "0.1",
                "--des", "experimento_1"
            ],
            "console": "integratedTerminal"
        },
        {
            "name": "DFGCN - Solo Evaluación",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/run.py",
            "args": [
                "--is_training", "0",
                "--model_id", "COP_USD",
                "--model", "DFGCN",
                "--data", "custom",
                "--root_path", "./datos/",
                "--data_path", "tasa_cop_usd.csv",
                "--features", "M",
                "--target", "COP_USD",
                "--freq", "d",
                "--seq_len", "96",
                "--label_len", "48",
                "--pred_len", "30",
                "--enc_in", "7",
                "--d_model", "128",
                "--n_heads", "1",
                "--e_layers", "1",
                "--d_ff", "128",
                "--patch_len", "8",
                "--k", "2",
                "--use_norm", "1",
                "--des", "experimento_1"
            ],
            "console": "integratedTerminal"
        },
        {
            "name": "Random Walk - Simulación",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/modelos/RandomWalk.py",
            "args": [
                "--data_path", "datos/tasa_cop_usd.csv",
                "--target_col", "COP_USD",
                "--pred_len", "30",
                "--num_simulations", "1000"
            ],
            "console": "integratedTerminal"
        }
    ]
}
```

4. Para ejecutar, presione `F5` o vaya a `Run > Start Debugging`
5. Seleccione la configuración deseada en el menú desplegable

### Visualizar resultados

Los gráficos de predicción se guardan automáticamente en:
- `./test_results/<nombre_experimento>/` — imágenes `.pdf` cada 200 batches
- `./results/<nombre_experimento>/` — matrices numpy con predicciones y valores reales
- `result_long_term_forecast.txt` — registro de métricas por experimento

---

## Referencia

El modelo DFGCN está basado en el repositorio original:
- **Repositorio**: https://github.com/junjieyeys/DFGCN
- **Artículo**: "DFGCN: Dual Frequency Graph Convolutional Network for Multivariate Time Series Forecasting"
