# Estudio de Caso: Cuantización y Propagación del Error en Traducción en Tiempo Real

Estudio de caso de INF-321 (Computación Numérica, UCM) que mide cómo la cuantización afecta la
propagación del error en una cascada **ASR → traducción con LLM**. El enunciado y la rúbrica
completos están en [`estudio_casos_unidad1-2.pdf`](estudio_casos_unidad1-2.pdf)
(fuente en `estudio_casos_unidad1-2.tex`).

El pipeline tiene dos etapas independientes, ambas en Julia sin dependencias externas:

1. **`asr_pipeline.jl`** — transcribe los audios de `audio/` con `whisper-cli` y calcula
   **WER**, **CER** y **RTF** contra las referencias de `groundtruth.txt`. Escribe las
   transcripciones en `out/`.
2. **`translate_pipeline.jl`** — traduce las transcripciones de `out/` con un modelo servido en
   Ollama y calcula **chrF**. Traduce tanto la referencia (`groundtruth.txt`) como la hipótesis
   del ASR con el *mismo* modelo, para aislar el error introducido específicamente por el ASR.
   Escribe las traducciones en `translations/`.

## Requisitos

- [Julia](https://julialang.org/) ≥ 1.10 (probado con 1.12).
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp) compilado (binario `whisper-cli`), con
  al menos un modelo `ggml-*.bin` descargado. **No está incluido en este repo** (ver más abajo).
- Un servidor [Ollama](https://ollama.com) accesible por HTTP con el modelo de traducción que
  quieras evaluar ya descargado (`ollama pull <modelo>`).
- `curl` disponible en el `PATH` (usado por `translate_pipeline.jl` para hablar con la API de
  Ollama).

## Preparar whisper.cpp

whisper.cpp es una dependencia externa grande (~2GB con modelos y entorno) y se mantiene fuera de
este repositorio. Clónalo y compílalo aparte, en la raíz del proyecto:

```bash
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build
cmake --build build -j --config Release

# descarga al menos un modelo (por ejemplo, base)
./models/download-ggml-model.sh base
cd ..
```

Esto deja el binario en `whisper.cpp/build/bin/whisper-cli` y el modelo en
`whisper.cpp/models/ggml-base.bin`, que es lo que esperan las rutas por defecto de
`asr_pipeline.jl`.

### Descargar modelos

`models/download-ggml-model.sh` descarga modelos ggml precompilados desde Hugging Face a
`whisper.cpp/models/`. Ejecuta el script sin argumentos para ver el listado completo de tamaños
disponibles:

```bash
cd whisper.cpp
./models/download-ggml-model.sh
```

Tamaños principales: `tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`, `large-v3`,
`large-v3-turbo`. Cada uno tiene variante `.en` (solo inglés) y, para varios tamaños, hay
versiones **ya cuantizadas** publicadas directamente (sufijo `-q5_0`, `-q5_1` o `-q8_0`), por
ejemplo:

```bash
./models/download-ggml-model.sh base            # fp16, línea base
./models/download-ggml-model.sh base-q5_1        # cuantizado Q5_1, precompilado
./models/download-ggml-model.sh base-q8_0        # cuantizado Q8_0, precompilado
```

Cada descarga queda en `whisper.cpp/models/ggml-<modelo>.bin`.

### Cuantizar modelos manualmente

Para generar tus propias variantes cuantizadas (necesario si quieres un tipo que no viene
precompilado, como `Q4_0`, o comparar varios niveles bajo el mismo commit de whisper.cpp para el
experimento de la rúbrica), usa el binario `whisper-quantize` que se compila junto con
`whisper-cli`:

```bash
# uso: whisper-quantize <modelo-origen.bin> <modelo-destino.bin> <tipo>
./whisper.cpp/build/bin/whisper-quantize \
  whisper.cpp/models/ggml-base.bin \
  whisper.cpp/models/ggml-base-q5_0.bin \
  q5_0
```

Tipos de cuantización soportados: `q2_k`, `q3_k`, `q4_0`, `q4_1`, `q4_k`, `q5_0`, `q5_1`,
`q5_k`, `q6_k`, `q8_0`. Para replicar la comparación pedida en la rúbrica (`Q8_0`, `Q5_1`,
`Q4_0`), genera las tres variantes a partir del mismo modelo fp16 base:

```bash
for t in q8_0 q5_1 q4_0; do
  ./whisper.cpp/build/bin/whisper-quantize \
    whisper.cpp/models/ggml-base.bin \
    whisper.cpp/models/ggml-base-$t.bin \
    $t
done
```

Luego corre `asr_pipeline.jl` una vez por cada `.bin` (original + cuantizados) para comparar
WER/CER/RTF entre niveles de cuantización.

## Ejecutar el pipeline

Desde la raíz del repositorio:

```bash
# 1. ASR: transcribe audio/test{1,2,3}.wav y reporta WER/CER/RTF
julia asr_pipeline.jl whisper.cpp/models/ggml-base.bin

# 2. Traducción: traduce out/*.txt vía Ollama y reporta chrF
#    <modelo_ollama> debe existir en el servidor Ollama configurado
julia translate_pipeline.jl gemma3:12b English
```

`translate_pipeline.jl` debe ejecutarse **después** de `asr_pipeline.jl`, porque lee las
transcripciones desde `out/`.

Por defecto, `translate_pipeline.jl` apunta a `http://geoespacial.ucm.cl:11434/api/generate`
(constante `OLLAMA_URL` en el script). Si usas otro servidor Ollama, edita esa constante o levanta
uno local (`ollama serve`, típicamente en `http://localhost:11434/api/generate`).

### Formato de `groundtruth.txt`

Una línea por archivo de audio, con el nombre del archivo y la transcripción de referencia entre
comillas:

```
test1.wav,"En tiempos de engaño universal, decir la verdad se convierte en un acto revolucionario"
```

### Salidas

- `out/<nombre>.txt` — transcripción generada por whisper-cli para cada audio.
- `translations/<nombre>_ref.txt` — traducción de la referencia (groundtruth).
- `translations/<nombre>_hyp.txt` — traducción de la hipótesis (salida del ASR).

Ambos scripts imprimen una tabla de métricas por archivo y el promedio al final.

## Reproducibilidad

Según la rúbrica del estudio de caso, todo resultado reportado debe indicar: hardware (CPU/GPU,
sistema operativo), *commit hash* de whisper.cpp, versión de Ollama, nombre y *tag* exacto de cada
modelo (incluyendo cuantización, ej. `Q8_0`/`Q5_1`/`Q4_0`), número de hilos, semillas y parámetros
de decodificación utilizados. Registra estos datos junto con cada corrida.
