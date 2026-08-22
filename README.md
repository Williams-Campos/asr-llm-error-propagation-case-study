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

Las transcripciones de referencia en inglés se encuentran en `groundtruth.txt` y las traducciones de referencia en español en `groundtruth_es.txt`.

---

## Estructura del Repositorio

```
.
├── README.md                          # Este archivo
├── PAUTA.md                           # Rúbrica de evaluación del estudio de caso
├── Especificación.txt                 # Especificación completa del entorno experimental
├── Especificación_por_modelo          # Representación numérica detallada por formato
├── Resultados.xlsx                    # Tabla consolidada de resultados
│
├── estudio_casos_unidad1-2.tex        # Informe LaTeX del estudio de caso
├── estudio_casos_unidad1-2.pdf        # Informe compilado (PDF)
│
├── audio/                             # Corpus: 10 archivos WAV (inglés, 15-30s c/u)
├── groundtruth.txt                    # Transcripciones de referencia (inglés)
├── groundtruth_es.txt                 # Traducciones de referencia (español)
│
├── asr_pipeline.jl                    # Pipeline ASR básico (uso interactivo)
├── translate_pipeline.jl              # Pipeline de traducción básico (uso interactivo)
│
├── benchmark_asr.jl                   # Benchmark R1/R2: ASR completo (5 formatos × 10 audios × 5 reps)
├── benchmark_translate.jl             # Benchmark R2: traducción (5 formatos ASR → LLM fijo)
├── benchmark_r3.jl                    # Benchmark R3: propagación del error en cascada
├── benchmark_weight_error.py          # Benchmark R1: error de cuantización peso a peso
├── benchmark_threads.py               # Benchmark R4: efecto del número de hilos
├── benchmark_reduction_order.py       # Benchmark R4: efecto del orden de reducción
├── medir_ram.sh                       # Script auxiliar: medición de RAM (RSS) por formato
│
├── grafico_figura1.py                 # Generador de Figura 1: precisión vs. bits por peso
├── grafico_figura2.py                 # Generador de Figura 2: error de representación (MAE)
├── grafico_figura3.py                 # Generador de Figura 3: propagación del error ASR→LLM
│
├── figura_1_precision_bits.{png,svg}  # Figura 1 generada
├── figura_2_error_representacion.{png,svg}  # Figura 2 generada
├── figura_3_propagacion_asr_llm.{png,svg}   # Figura 3 generada
│
├── asr_per_audio.csv                  # Resultados brutos ASR por audio/formato/repetición
├── quantization_error_summary.csv     # Resumen de error de cuantización por formato
├── quantization_error_by_tensor.csv   # Error de cuantización desglosado por tensor
├── translation_per_audio.csv          # Resultados de traducción R2 por audio
├── translation_summary.csv            # Resumen de traducción R2
├── translation_metadata.csv           # Metadatos del experimento R2
├── r3_per_audio.csv                   # Resultados de propagación R3 por audio
├── r3_summary.csv                     # Resumen de propagación R3
├── r3_metadata.csv                    # Metadatos del experimento R3
├── thread_experiment.csv              # Resultados brutos del experimento de hilos
├── thread_summary.csv                 # Resumen del experimento de hilos
├── reduction_order_experiment.csv     # Resultados del experimento de orden de reducción
├── ram_resultados.csv                 # Mediciones de RAM (RSS) por formato
│
├── results/                           # Resultados auxiliares del benchmark ASR
│   ├── asr_metadata.csv
│   └── asr_summary.csv
├── out/                               # Transcripciones generadas por whisper-cli
│   └── asr/{F32,F16,Q8_0,Q5_1,Q4_0}/run_XX/  # Organizadas por formato y repetición
├── out_threads/                       # Transcripciones del experimento de hilos
├── translations/                      # Traducciones generadas por Ollama
│   └── asr/{F32,F16,Q8_0,Q5_1,Q4_0}/run_XX/  # Organizadas por formato y repetición
│
├── whisper.cpp/                       # (git-ignored) Clonar externamente
└── whisper/                           # (git-ignored) OpenAI Whisper (para conversión F32)
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

### `asr_pipeline.jl` — Pipeline ASR básico

Pipeline interactivo de transcripción. Transcribe los 10 archivos WAV de `audio/` usando `whisper-cli` y calcula métricas contra `groundtruth.txt`.

```bash
julia asr_pipeline.jl whisper.cpp/models/ggml-base-f32.bin
```

**Métricas calculadas:** WER (Word Error Rate), CER (Character Error Rate), RTF (Real-Time Factor).  
**Salida:** Transcripciones en `out/`, tabla de métricas por audio y promedio en la terminal.

### `translate_pipeline.jl` — Pipeline de traducción básico

Traduce las transcripciones de `out/` usando un LLM servido en Ollama y calcula chrF comparando contra la traducción de referencia generada por el mismo modelo.

```bash
julia translate_pipeline.jl gemma3:1b-it-q8_0 English
```

**Métricas calculadas:** chrF (Character F-score).  
**Salida:** Traducciones en `translations/`, tabla de métricas en la terminal.

> **Nota:** `translate_pipeline.jl` debe ejecutarse **después** de `asr_pipeline.jl`, ya que lee las transcripciones desde `out/`.

---

## Benchmarks Experimentales

### `benchmark_asr.jl` — Benchmark ASR (R1/R2)

**Propósito:** Ejecutar el experimento completo de transcripción con los 5 formatos de cuantización del modelo Whisper Base, midiendo el efecto de la precisión numérica sobre la calidad del ASR.

**Diseño experimental:**
- 5 formatos: F32, F16, Q8_0, Q5_1, Q4_0
- 10 audios del corpus
- 5 repeticiones por combinación
- Total: 250 transcripciones

**Parámetros fijos:**
- Idioma: `en`
- Hilos: 4
- Beam size: 5
- Best-of: 5
- Temperatura: 0.0
- Incremento de temperatura: 0.0
- Fallback de temperatura: deshabilitado (`-nf`)

```bash
julia benchmark_asr.jl          # 5 repeticiones (por defecto)
julia benchmark_asr.jl 10       # 10 repeticiones
```

**Salidas:**
- `asr_per_audio.csv` — Resultados por audio/formato/repetición (WER, CER, RTF, texto)
- `results/asr_summary.csv` — Promedios por formato
- `results/asr_metadata.csv` — Metadatos del experimento

---

### `benchmark_translate.jl` — Benchmark de traducción (R2)

**Propósito:** Medir el efecto de la cuantización del ASR sobre la calidad de la traducción, manteniendo el LLM fijo en `gemma3:1b-it-q8_0`. Esto aísla el error introducido específicamente por el ASR.

**Diseño experimental:**
- LLM fijo: `gemma3:1b-it-q8_0` (Q8_0)
- 5 condiciones de entrada ASR: F32, F16, Q8_0, Q5_1, Q4_0
- 10 audios del corpus
- Traducción comparada contra `groundtruth_es.txt`
- Temperatura LLM: 0.0, semilla: 42

```bash
julia benchmark_translate.jl
```

**Salidas:**
- `translation_per_audio.csv` — chrF por audio y formato ASR
- `translation_summary.csv` — Promedios por formato
- `translation_metadata.csv` — Parámetros del experimento
- `translations/asr/*/` — Archivos de traducción (hipótesis `_hyp.txt` y referencia `_ref.txt`)

---

### `benchmark_r3.jl` — Propagación del error en cascada (R3)

**Propósito:** Separar y cuantificar las fuentes de error en la cascada ASR → LLM. Se evalúan dos cuantizaciones del **mismo LLM** (Gemma 3 1B) para aislar el efecto numérico del modelo de lenguaje.

**Diseño experimental — tres condiciones por LLM:**

| Condición | Entrada | Mide |
|---|---|---|
| `llm_only` | Transcripción de referencia → LLM | Error del LLM aislado |
| `asr_high_precision` | Salida ASR F32 → LLM | Error ASR alta precisión + LLM |
| `asr_quantized` | Salida ASR cuantizado → LLM | Error compuesto (ASR + LLM) |

**Combinaciones evaluadas:**
- 2 LLM (Q8_0, Q4_K_M) × 5 condiciones (Referencia, F32, Q8_0, Q5_1, Q4_0) × 10 audios = **100 traducciones**

```bash
julia benchmark_r3.jl
```

**Salidas:**
- `r3_per_audio.csv` — chrF por audio/LLM/condición
- `r3_summary.csv` — Promedios por condición y modelo
- `r3_metadata.csv` — Parámetros del experimento

---

### `benchmark_weight_error.py` — Error de cuantización peso a peso (R1)

**Propósito:** Medir el error de representación introducido por cada formato de cuantización, comparando directamente los pesos reconstruidos de cada modelo cuantizado contra los pesos originales en F32. Esto verifica experimentalmente la cota teórica de cuantización |εᵢ| ≤ d/2.

**Metodología:**
- Lee los tensores de cada archivo `.bin` en formato GGML legacy (Whisper)
- Decodifica los bloques cuantizados (Q8_0, Q5_1, Q4_0) y los reconstruye a F32
- Calcula: MAE, error mediano, error máximo, RMSE, error relativo, porcentaje de pesos dentro de la cota d/2

```bash
python benchmark_weight_error.py
```

**Salidas:**
- `quantization_error_summary.csv` — Estadísticas globales de error por formato
- `quantization_error_by_tensor.csv` — Error desglosado por cada tensor del modelo

---

### `benchmark_threads.py` — Efecto del número de hilos (R4)

**Propósito:** Evaluar cómo el número de hilos de ejecución del CPU afecta la latencia y la estabilidad de las transcripciones de whisper.cpp. Evidencia el efecto del paralelismo sobre el resultado numérico.

**Diseño experimental:**
- 2 modelos: F32, Q4_0
- 4 configuraciones de hilos: 1, 2, 4, 8
- 3 audios seleccionados del corpus
- 3 repeticiones por combinación
- Compara si las transcripciones difieren al cambiar hilos

```bash
python benchmark_threads.py
```

**Salidas:**
- `thread_experiment.csv` — Resultados brutos (RTF, tiempo, hash de transcripción)
- `thread_summary.csv` — Resumen con RTF promedio y conteo de transcripciones diferentes
- `out_threads/` — Transcripciones generadas por cada combinación

---

### `benchmark_reduction_order.py` — Efecto del orden de reducción (R4)

**Propósito:** Demostrar que el orden en que se suman los pesos de un tensor afecta el resultado numérico debido a la aritmética de punto flotante. Se extrae un tensor específico del modelo F32 y se comparan distintas estrategias de suma.

**Metodología:**
- Tensor evaluado: `encoder.blocks.0.attn.query.weight`
- Estrategias: suma secuencial, inversa, por magnitud ascendente/descendente, suma de Kahan, suma por pares, orden aleatorio
- Compara cada resultado contra la suma de Kahan (referencia de alta precisión)

```bash
python benchmark_reduction_order.py
```

**Salida:**
- `reduction_order_experiment.csv` — Resultado de cada estrategia, error absoluto y relativo respecto a Kahan

---

### `medir_ram.sh` — Medición de RAM por formato

**Propósito:** Medir el consumo máximo de memoria (Maximum Resident Set Size) de whisper-cli para cada formato del modelo, usando `/usr/bin/time -l`.

```bash
bash medir_ram.sh
```

**Salida:**
- `ram_resultados.csv` — RSS en bytes y MiB por formato y repetición

**Resultados observados:**

| Formato | RSS máximo |
|---|---|
| F32 | 526,38 MiB |
| F16 | 341,94 MiB |
| Q8_0 | 255,70 MiB |
| Q5_1 | 227,42 MiB |
| Q4_0 | 208,91 MiB |

---

## Generación de Figuras

Las tres figuras principales del informe se generan con scripts Python que usan `matplotlib`. Cada script produce versiones PNG y SVG:

### `grafico_figura1.py` — Figura 1: Precisión vs. Bits por Peso

Grafica la curva de **WER** (tasa de error de palabras) y **chrF** (calidad de traducción) en función de los bits efectivos por peso de cada formato de cuantización. Permite identificar visualmente el punto de quiebre donde la degradación crece de forma no lineal.

```bash
python grafico_figura1.py
# → figura_1_precision_bits.png, figura_1_precision_bits.svg
```

### `grafico_figura2.py` — Figura 2: Error de Representación

Grafica el **error absoluto medio (MAE)** de los pesos reconstruidos respecto a los pesos F32 originales para cada formato cuantizado (F16, Q8_0, Q5_1, Q4_0). Muestra cómo el error de representación crece al reducir la precisión.

```bash
python grafico_figura2.py
# → figura_2_error_representacion.png, figura_2_error_representacion.svg
```

### `grafico_figura3.py` — Figura 3: Propagación del Error en la Cascada ASR → LLM

Grafica la **chrF** obtenida bajo cada condición experimental del R3, comparando los dos modelos LLM (Gemma Q8_0 vs. Gemma Q4_K_M). Visualiza cómo el error del ASR se propaga y amplifica al pasar por el LLM.

```bash
python grafico_figura3.py
# → figura_3_propagacion_asr_llm.png, figura_3_propagacion_asr_llm.svg
```

---

## Resultados (archivos CSV)

| Archivo | Descripción |
|---|---|
| `asr_per_audio.csv` | WER, CER, RTF por audio, formato y repetición (250 filas) |
| `results/asr_summary.csv` | Promedios del benchmark ASR por formato |
| `results/asr_metadata.csv` | Metadatos del experimento ASR |
| `quantization_error_summary.csv` | MAE, RMSE, error relativo, cota d/2 por formato |
| `quantization_error_by_tensor.csv` | Error por tensor individual del modelo (~99 tensores) |
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

Las transcripciones y traducciones generadas se almacenan organizadas por formato de cuantización y número de repetición:

```
out/asr/{F32,F16,Q8_0,Q5_1,Q4_0}/run_{01..05}/
    audio01(dolor toracico).txt
    audio02(alergia).txt
    ...
    audio10(dolor cabeza).txt

translations/asr/{F32,F16,Q8_0,Q5_1,Q4_0}/run_{01..05}/
    audio01(dolor toracico)_hyp.txt    # Traducción de la hipótesis ASR
    audio01(dolor toracico)_ref.txt    # Traducción de la referencia
    ...

out_threads/
    F32_audio01(dolor toracico)_t{1,2,4,8}_r{1,2,3}.txt
    Q4_0_audio01(dolor toracico)_t{1,2,4,8}_r{1,2,3}.txt
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

Trabajo desarrollado para la asignatura **INF-321 Computación Numérica** de la carrera de Ingeniería Civil Informática, Universidad Católica del Maule. La rúbrica completa se encuentra en [`PAUTA.md`](PAUTA.md) y el enunciado original en [`estudio_casos_unidad1-2.pdf`](estudio_casos_unidad1-2.pdf).
