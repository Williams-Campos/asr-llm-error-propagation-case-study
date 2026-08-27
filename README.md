# Cuantización y Propagación del Error en Traducción en Tiempo Real

Estudio de caso para **INF-321 — Computación Numérica** (Universidad Católica del Maule).  
Evalúa experimentalmente cómo la cuantización de modelos afecta la precisión y la propagación del error en una cascada de **reconocimiento automático del habla (ASR) → traducción con modelo de lenguaje (LLM)**, ejecutada completamente en el dispositivo local.

---

## Tabla de Contenidos

1. [Descripción del Problema](#descripción-del-problema)  
2. [Arquitectura del Sistema](#arquitectura-del-sistema)  
3. [Entorno Experimental](#entorno-experimental)  
4. [Obtención de los Modelos](#obtención-de-los-modelos)  
5. [Corpus de Audio](#corpus-de-audio)  
6. [Estructura del Repositorio](#estructura-del-repositorio)  
7. [Requisitos e Instalación](#requisitos-e-instalación)  
8. [Scripts de Pipeline](#scripts-de-pipeline)  
9. [Benchmarks Experimentales](#benchmarks-experimentales)  
10. [Generación de Figuras](#generación-de-figuras)  
11. [Resultados (archivos CSV)](#resultados-archivos-csv)  
12. [Salidas de Ejecución](#salidas-de-ejecución)  
13. [Reproducibilidad](#reproducibilidad)  
14. [Licencia y Créditos](#licencia-y-créditos)

---

## Descripción del Problema

Un centro de atención de urgencia necesita un sistema de traducción en tiempo real para atender pacientes que no hablan español. Por confidencialidad de los datos clínicos, el sistema debe ejecutarse **completamente on-device** (sin APIs en la nube), con **latencia inferior al tiempo real** (RTF < 1).

El equipo disponible no tiene memoria suficiente para ambos modelos en precisión completa, por lo que se recurre a la **cuantización**: los pesos originales en punto flotante de 32 bits (IEEE-754 binary32) se convierten a formatos de menor ancho de palabra. Esto introduce un **error de representación** que se propaga a través de la cascada.

**Pregunta central:**  
> *¿Cuánta precisión numérica se puede sacrificar antes de que la traducción deje de ser clínicamente confiable?*

---

## Arquitectura del Sistema

La arquitectura es una cascada de dos etapas:

```
Audio (WAV) ──► whisper.cpp (ASR) ──► Texto inglés ──► Ollama/Gemma 3 (LLM) ──► Texto español
                   │                                        │
              Formatos:                                Cuantizaciones:
              F32, F16, Q8_0,                          Q8_0 (gemma3:1b-it-q8_0)
              Q5_1, Q4_0                               Q4_K_M (gemma3:1b)
```

1. **Etapa ASR**: [whisper.cpp](https://github.com/ggml-org/whisper.cpp) transcribe el audio en inglés a texto.
2. **Etapa LLM**: [Ollama](https://ollama.com) con Gemma 3 1B traduce el texto inglés al español.

---

## Entorno Experimental

| Componente | Detalle |
|---|---|
| **Hardware** | MacBook Air (M1, 2020), 8 GB RAM unificada, SSD 256 GB |
| **CPU** | Apple M1 — 4 núcleos de rendimiento + 4 de eficiencia |
| **GPU** | GPU integrada Apple M1 (7 núcleos, Metal) |
| **Sistema operativo** | macOS Tahoe 26.5.2 |
| **Julia** | 1.12.6 |
| **whisper.cpp** | v1.9.2 (commit `592feef04a1802b18cbeffd0fd0eb5d02570c2ec`) |
| **Ollama** | 0.32.2 |
| **Python** | 3.14 (para scripts de gráficos y benchmarks auxiliares) |

---

## Obtención de los Modelos

### Modelo ASR — Whisper Base

El modelo de referencia en precisión completa (**F32**) se obtuvo a partir del repositorio oficial de [OpenAI Whisper](https://github.com/openai/whisper). Whisper es un modelo de reconocimiento automático del habla desarrollado por OpenAI, entrenado en 680.000 horas de audio multilingüe. Se utilizó la variante **Base** (74 M de parámetros).

El modelo F32 se convirtió al formato GGML compatible con whisper.cpp usando el script de conversión incluido en whisper.cpp (`convert-pt-to-ggml.py`), obteniendo `ggml-base-f32.bin` (291 MB).

A partir del modelo F32 de referencia, se generaron las versiones cuantizadas con la herramienta `whisper-quantize`:

| Formato | Bits efectivos/peso | Tamaño en disco | Representación |
|---|---|---|---|
| **F32** | 32,0 | 291 MB | IEEE-754 binary32 |
| **F16** | 16,0 | 148 MB | IEEE-754 binary16 |
| **Q8_0** | 8,5 | 81,77 MB | Bloques de 32 pesos × 8 bits + escala FP16 |
| **Q5_1** | 6,0 | 59,71 MB | Bloques de 32 pesos × 5 bits + escala FP16 + offset FP16 |
| **Q4_0** | 4,5 | 46,47 MB | Bloques de 32 pesos × 4 bits + escala FP16 |

```bash
# Cuantización manual a partir del modelo F32
for t in q8_0 q5_1 q4_0; do
  ./whisper.cpp/build/bin/whisper-quantize \
    whisper.cpp/models/ggml-base-f32.bin \
    whisper.cpp/models/ggml-base-$t.bin \
    $t
done
```

### Modelo LLM — Gemma 3 1B

Se utilizó [Gemma 3](https://ai.google.dev/gemma) de Google DeepMind (999,89 M de parámetros) servido con Ollama, en dos cuantizaciones:

| Tag en Ollama | Cuantización | Tamaño | ID |
|---|---|---|---|
| `gemma3:1b-it-q8_0` | Q8_0 | 1,1 GB | `0fdb9c7fefee` |
| `gemma3:1b` | Q4_K_M | 815 MB | `8648f39daa8f` |

---

## Corpus de Audio

El corpus consta de **10 segmentos de audio** en inglés, cada uno de 15–30 segundos de duración, que simulan relatos clínicos de pacientes en un servicio de urgencia. La duración total del corpus es de **212,288 s**.

| # | Archivo | Escenario clínico |
|---|---|---|
| 1 | `audio01(dolor toracico).wav` | Dolor torácico severo |
| 2 | `audio02(alergia).wav` | Alergia a penicilina |
| 3 | `audio03(accidente).wav` | Accidente en bicicleta |
| 4 | `audio04(dolor abdominal).wav` | Dolor abdominal |
| 5 | `audio05(diabetes).wav` | Diabetes tipo 2 |
| 6 | `audio06(fiebre).wav` | Fiebre alta |
| 7 | `audio07(respiracion).wav` | Dificultad respiratoria |
| 8 | `audio08(herida).wav` | Herida cortante en mano |
| 9 | `audio09(mareo).wav` | Mareo súbito |
| 10 | `audio10(dolor cabeza).wav` | Dolor de cabeza severo |

Las transcripciones de referencia en inglés se encuentran en `data/groundtruth.txt` y las traducciones de referencia en español en `data/groundtruth_es.txt`.

---

## Estructura del Repositorio

```
.
├── README.md                              # Este archivo
├── .gitignore                             # Reglas de exclusión para Git
│
├── docs/                                  # Documentación del proyecto
│   ├── PAUTA.md                           # Rúbrica de evaluación del estudio de caso
│   ├── especificacion_entorno.txt         # Especificación completa del entorno experimental
│   ├── especificacion_modelos.txt         # Representación numérica detallada por formato
│   └── informe/                           # Informe LaTeX
│       ├── estudio_casos_unidad1-2.tex
│       └── estudio_casos_unidad1-2.pdf
│
├── data/                                  # Datos de entrada
│   ├── audio/                             # Corpus: 10 archivos WAV (inglés, 15-30s c/u)
│   ├── groundtruth.txt                    # Transcripciones de referencia (inglés)
│   └── groundtruth_es.txt                 # Traducciones de referencia (español)
│
├── scripts/                               # Todos los scripts ejecutables
│   ├── pipelines/                         # Pipelines interactivos
│   │   ├── asr_pipeline.jl                # Pipeline ASR básico
│   │   └── translate_pipeline.jl          # Pipeline de traducción básico
│   ├── benchmarks/                        # Scripts de benchmarking
│   │   ├── benchmark_asr.jl               # Benchmark R1/R2: ASR completo
│   │   ├── benchmark_translate.jl         # Benchmark R2: traducción
│   │   ├── benchmark_r3.jl               # Benchmark R3: propagación del error
│   │   ├── benchmark_weight_error.py      # Benchmark R1: error de cuantización
│   │   ├── benchmark_threads.py           # Benchmark R4: efecto de hilos
│   │   ├── benchmark_reduction_order.py   # Benchmark R4: orden de reducción
│   │   └── medir_ram.sh                   # Medición de RAM (RSS) por formato
│   └── figures/                           # Generadores de figuras
│       ├── grafico_figura1.py             # Figura 1: precisión vs. bits por peso
│       ├── grafico_figura2.py             # Figura 2: error de representación
│       └── grafico_figura3.py             # Figura 3: propagación del error ASR→LLM
│
├── results/                               # Todos los resultados experimentales
│   ├── csv/                               # Datos tabulares
│   │   ├── asr_per_audio.csv              # WER, CER, RTF por audio/formato/repetición
│   │   ├── asr_summary.csv               # Promedios del benchmark ASR
│   │   ├── asr_metadata.csv              # Metadatos del experimento ASR
│   │   ├── quantization_error_summary.csv # MAE, RMSE, error relativo por formato
│   │   ├── quantization_error_by_tensor.csv # Error por tensor del modelo
│   │   ├── translation_per_audio.csv      # chrF por audio y formato ASR (R2)
│   │   ├── translation_summary.csv        # Promedios de traducción R2
│   │   ├── translation_metadata.csv       # Parámetros del experimento R2
│   │   ├── r3_per_audio.csv              # chrF por audio/LLM/condición (R3)
│   │   ├── r3_summary.csv               # Promedios de propagación R3
│   │   ├── r3_metadata.csv              # Parámetros del experimento R3
│   │   ├── thread_experiment.csv         # RTF bruto por modelo/hilos/audio
│   │   ├── thread_summary.csv            # Resumen del efecto de hilos
│   │   ├── reduction_order_experiment.csv # Error por estrategia de reducción
│   │   └── ram_resultados.csv            # RSS por formato y repetición
│   ├── figures/                           # Figuras generadas (PNG y SVG)
│   │   ├── figura_1_precision_bits.*
│   │   ├── figura_2_error_representacion.*
│   │   └── figura_3_propagacion_asr_llm.*
│   └── Resultados.xlsx                    # Tabla consolidada de resultados
│
├── output/                                # Salidas crudas de ejecución
│   ├── transcriptions/                    # Transcripciones ASR por formato y repetición
│   │   └── {F32,F16,Q8_0,Q5_1,Q4_0}/run_XX/
│   ├── translations/                      # Traducciones LLM (_hyp.txt y _ref.txt)
│   │   └── {F32,F16,Q8_0,Q5_1,Q4_0}/run_XX/
│   └── threads/                           # Transcripciones del experimento de hilos
│
├── whisper.cpp/                           # (git-ignored) Clonar externamente
└── whisper/                               # (git-ignored) OpenAI Whisper (para conversión F32)
```

---

## Requisitos e Instalación

### Dependencias

- [Julia](https://julialang.org/) ≥ 1.10 (probado con 1.12.6)
- [Python](https://www.python.org/) ≥ 3.10 con `numpy` y `matplotlib`
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) compilado localmente
- [Ollama](https://ollama.com) con los modelos Gemma 3 descargados
- `curl` en el PATH (usado por los scripts de traducción)

### Compilar whisper.cpp

```bash
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
git checkout 592feef04a1802b18cbeffd0fd0eb5d02570c2ec  # v1.9.2
cmake -B build
cmake --build build -j --config Release
cd ..
```

### Obtener el modelo F32 desde OpenAI Whisper

El modelo F32 (**ggml-base-f32.bin**) se obtuvo convirtiendo los pesos originales del repositorio de [OpenAI Whisper](https://github.com/openai/whisper) al formato GGML:

```bash
# Clonar OpenAI Whisper
git clone https://github.com/openai/whisper
cd whisper
pip install -e .

# Descargar el modelo Base de OpenAI (se descarga automáticamente al primer uso)
python -c "import whisper; whisper.load_model('base')"

# Convertir a formato GGML F32 usando el script de whisper.cpp
cd ../whisper.cpp
python models/convert-pt-to-ggml.py ~/.cache/whisper/base.pt ../whisper/ models/
mv models/ggml-model.bin models/ggml-base-f32.bin
cd ..
```

### Descargar el modelo F16 (línea base precompilada)

```bash
cd whisper.cpp
./models/download-ggml-model.sh base   # descarga ggml-base.bin (F16)
mv models/ggml-base.bin models/ggml-base-f16.bin
cd ..
```

### Generar las variantes cuantizadas

```bash
cd whisper.cpp
for t in q8_0 q5_1 q4_0; do
  ./build/bin/whisper-quantize \
    models/ggml-base-f32.bin \
    models/ggml-base-$t.bin \
    $t
done
cd ..
```

### Preparar Ollama

```bash
ollama pull gemma3:1b-it-q8_0   # Gemma 3 1B en Q8_0
ollama pull gemma3:1b            # Gemma 3 1B en Q4_K_M
ollama serve                     # Iniciar el servidor (si no está corriendo)
```

---

## Scripts de Pipeline

### `scripts/pipelines/asr_pipeline.jl` — Pipeline ASR básico

Pipeline interactivo de transcripción. Transcribe los 10 archivos WAV de `data/audio/` usando `whisper-cli` y calcula métricas contra `data/groundtruth.txt`.

```bash
julia scripts/pipelines/asr_pipeline.jl whisper.cpp/models/ggml-base-f32.bin
```

**Métricas calculadas:** WER (Word Error Rate), CER (Character Error Rate), RTF (Real-Time Factor).  
**Salida:** Transcripciones en `output/transcriptions/`, tabla de métricas por audio y promedio en la terminal.

### `scripts/pipelines/translate_pipeline.jl` — Pipeline de traducción básico

Traduce las transcripciones de `output/transcriptions/` usando un LLM servido en Ollama y calcula chrF comparando contra la traducción de referencia generada por el mismo modelo.

```bash
julia scripts/pipelines/translate_pipeline.jl gemma3:1b-it-q8_0 English
```

**Métricas calculadas:** chrF (Character F-score).  
**Salida:** Traducciones en `output/translations/`, tabla de métricas en la terminal.

> **Nota:** `translate_pipeline.jl` debe ejecutarse **después** de `asr_pipeline.jl`, ya que lee las transcripciones desde `output/transcriptions/`.

---

## Benchmarks Experimentales

### `scripts/benchmarks/benchmark_asr.jl` — Benchmark ASR (R1/R2)

**Propósito:** Ejecutar el experimento completo de transcripción con los 5 formatos de cuantización del modelo Whisper Base, midiendo el efecto de la precisión numérica sobre la calidad del ASR.

**Diseño experimental:**
- 5 formatos: F32, F16, Q8_0, Q5_1, Q4_0
- 10 audios del corpus
- 5 repeticiones por combinación
- Total: 250 transcripciones

```bash
julia scripts/benchmarks/benchmark_asr.jl          # 5 repeticiones (por defecto)
julia scripts/benchmarks/benchmark_asr.jl 10       # 10 repeticiones
```

**Salidas:**
- `results/csv/asr_per_audio.csv` — Resultados por audio/formato/repetición
- `results/csv/asr_summary.csv` — Promedios por formato
- `results/csv/asr_metadata.csv` — Metadatos del experimento

---

### `scripts/benchmarks/benchmark_translate.jl` — Benchmark de traducción (R2)

**Propósito:** Medir el efecto de la cuantización del ASR sobre la calidad de la traducción, manteniendo el LLM fijo en `gemma3:1b-it-q8_0`.

```bash
julia scripts/benchmarks/benchmark_translate.jl
```

**Salidas:**
- `results/csv/translation_per_audio.csv` — chrF por audio y formato ASR
- `results/csv/translation_summary.csv` — Promedios por formato
- `results/csv/translation_metadata.csv` — Parámetros del experimento
- `output/translations/` — Archivos de traducción

---

### `scripts/benchmarks/benchmark_r3.jl` — Propagación del error en cascada (R3)

**Propósito:** Separar y cuantificar las fuentes de error en la cascada ASR → LLM.

```bash
julia scripts/benchmarks/benchmark_r3.jl
```

**Salidas:**
- `results/csv/r3_per_audio.csv` — chrF por audio/LLM/condición
- `results/csv/r3_summary.csv` — Promedios por condición y modelo
- `results/csv/r3_metadata.csv` — Parámetros del experimento

---

### `scripts/benchmarks/benchmark_weight_error.py` — Error de cuantización peso a peso (R1)

**Propósito:** Medir el error de representación introducido por cada formato de cuantización, comparando directamente los pesos reconstruidos contra los pesos F32.

```bash
python scripts/benchmarks/benchmark_weight_error.py
```

**Salidas:**
- `results/csv/quantization_error_summary.csv` — Estadísticas globales de error por formato
- `results/csv/quantization_error_by_tensor.csv` — Error desglosado por tensor

---

### `scripts/benchmarks/benchmark_threads.py` — Efecto del número de hilos (R4)

**Propósito:** Evaluar cómo el número de hilos afecta la latencia y estabilidad de las transcripciones.

```bash
python scripts/benchmarks/benchmark_threads.py
```

**Salidas:**
- `results/csv/thread_experiment.csv` — RTF bruto por combinación
- `results/csv/thread_summary.csv` — Resumen con RTF promedio
- `output/threads/` — Transcripciones generadas

---

### `scripts/benchmarks/benchmark_reduction_order.py` — Efecto del orden de reducción (R4)

**Propósito:** Demostrar que el orden de suma de los pesos afecta el resultado numérico.

```bash
python scripts/benchmarks/benchmark_reduction_order.py
```

**Salida:**
- `results/csv/reduction_order_experiment.csv` — Resultado por estrategia de suma

---

### `scripts/benchmarks/medir_ram.sh` — Medición de RAM por formato

**Propósito:** Medir el consumo máximo de memoria (RSS) de whisper-cli para cada formato.

```bash
bash scripts/benchmarks/medir_ram.sh
```

**Salida:**
- `results/csv/ram_resultados.csv` — RSS en bytes y MiB por formato y repetición

---

## Generación de Figuras

Las tres figuras principales se generan con scripts Python en `scripts/figures/`. Cada script produce versiones PNG y SVG en `results/figures/`:

```bash
python scripts/figures/grafico_figura1.py   # → results/figures/figura_1_precision_bits.*
python scripts/figures/grafico_figura2.py   # → results/figures/figura_2_error_representacion.*
python scripts/figures/grafico_figura3.py   # → results/figures/figura_3_propagacion_asr_llm.*
```

---

## Resultados (archivos CSV)

Todos los resultados tabulares se encuentran en `results/csv/`:

| Archivo | Descripción |
|---|---|
| `asr_per_audio.csv` | WER, CER, RTF por audio, formato y repetición (250 filas) |
| `asr_summary.csv` | Promedios del benchmark ASR por formato |
| `asr_metadata.csv` | Metadatos del experimento ASR |
| `quantization_error_summary.csv` | MAE, RMSE, error relativo, cota d/2 por formato |
| `quantization_error_by_tensor.csv` | Error por tensor individual del modelo |
| `translation_per_audio.csv` | chrF por audio y formato ASR (benchmark R2) |
| `translation_summary.csv` | Promedios de traducción R2 por formato |
| `translation_metadata.csv` | Parámetros del experimento R2 |
| `r3_per_audio.csv` | chrF por audio/LLM/condición (benchmark R3) |
| `r3_summary.csv` | Promedios de propagación R3 por condición |
| `r3_metadata.csv` | Parámetros del experimento R3 |
| `thread_experiment.csv` | RTF bruto por modelo/hilos/audio/repetición |
| `thread_summary.csv` | Resumen del efecto de hilos sobre latencia |
| `reduction_order_experiment.csv` | Error de cada estrategia de reducción |
| `ram_resultados.csv` | RSS por formato y repetición |

---

## Salidas de Ejecución

Las transcripciones y traducciones generadas se almacenan en `output/`:

```
output/transcriptions/{F32,F16,Q8_0,Q5_1,Q4_0}/run_{01..05}/
    audio01(dolor toracico).txt
    ...
    audio10(dolor cabeza).txt

output/translations/{F32,F16,Q8_0,Q5_1,Q4_0}/run_{01..05}/
    audio01(dolor toracico)_hyp.txt    # Traducción de la hipótesis ASR
    audio01(dolor toracico)_ref.txt    # Traducción de la referencia
    ...

output/threads/
    F32_audio01(dolor toracico)_t{1,2,4,8}_r{1,2,3}.txt
    ...
```

---

## Reproducibilidad

Para reproducir los resultados de este estudio, se deben respetar los siguientes parámetros:

### Parámetros de decodificación ASR

| Parámetro | Valor |
|---|---|
| Idioma | `en` |
| Hilos | 4 |
| Beam size | 5 |
| Best-of | 5 |
| Temperatura | 0.0 |
| Incremento de temperatura | 0.0 |
| Fallback de temperatura | Deshabilitado (`-nf`) |
| GPU | Metal (Apple Silicon) |

### Parámetros del LLM (Ollama)

| Parámetro | Valor |
|---|---|
| Temperatura | 0.0 |
| Semilla | 42 |
| top_k | 64 |
| top_p | 0.95 |
| Streaming | `false` |

### Versiones exactas

| Software | Versión / Commit |
|---|---|
| whisper.cpp | v1.9.2 — `592feef04a1802b18cbeffd0fd0eb5d02570c2ec` |
| Ollama | 0.32.2 |
| Julia | 1.12.6 |
| Modelo Whisper | Base (74M parámetros) |
| Modelo LLM | Gemma 3 1B (`0fdb9c7fefee` Q8_0 / `8648f39daa8f` Q4_K_M) |

---

## Licencia y Créditos

- **Whisper** — [OpenAI](https://github.com/openai/whisper), licencia MIT. Se utilizó para obtener el modelo F32 de referencia.
- **whisper.cpp** — [ggml-org](https://github.com/ggml-org/whisper.cpp), licencia MIT. Implementación en C/C++ del modelo Whisper sobre la biblioteca ggml.
- **Gemma 3** — [Google DeepMind](https://ai.google.dev/gemma). Modelo de lenguaje servido mediante Ollama.
- **Ollama** — [ollama.com](https://ollama.com), licencia MIT.

Trabajo desarrollado para la asignatura **INF-321 Computación Numérica** de la carrera de Ingeniería Civil Informática, Universidad Católica del Maule. La rúbrica completa se encuentra en [`docs/PAUTA.md`](docs/PAUTA.md) y el enunciado original en [`docs/informe/estudio_casos_unidad1-2.pdf`](docs/informe/estudio_casos_unidad1-2.pdf).
