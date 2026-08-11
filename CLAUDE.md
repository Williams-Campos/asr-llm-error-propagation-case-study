# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Coursework for INF-321 (Computación Numérica, UCM) — a case study ("Cuantización y Propagación
del Error en Traducción en Tiempo Real") measuring how quantization affects error propagation
through an ASR → LLM translation cascade. The assignment brief and rubric are in
`estudio_casos_unidad1-2.tex` (compiles to the `.pdf` in the same directory). The actual
experiment is two standalone Julia scripts at the repo root; `whisper.cpp` is a vendored upstream
clone used only as the ASR engine (do not treat it as part of this project's own code — it has
its own `AGENTS.md` with a strict no-AI-PR policy that applies to *its own* upstream repo, not to
work done here).

## Pipeline stages

1. **ASR** (`asr_pipeline.jl`): runs `whisper.cpp/build/bin/whisper-cli` over each WAV in `audio/`
   against a given model, computes WER/CER against `groundtruth.txt`, and writes transcripts to
   `out/<basename>.txt`.
2. **Translation** (`translate_pipeline.jl`): reads transcripts from `out/`, sends both the
   reference (groundtruth) sentence and the ASR hypothesis through the *same* Ollama-served LLM
   for translation, and scores chrF between the two translations. Writes `translations/<basename>_ref.txt`
   and `translations/<basename>_hyp.txt`.

The design point of stage 2 is isolating ASR-induced error: reference and hypothesis both pass
through the identical translation model/prompt, so any quality gap is attributable to the ASR
transcription, not the LLM.

Both scripts share the `filename,"transcripcion"` format for `groundtruth.txt` (one entry per
line, comma before the opening quote, e.g. `test1.wav,"texto de referencia"`) and re-implement
their own primitives with zero external Julia packages — Levenshtein/WER/CER, WAV header parsing,
minimal JSON field extraction, and chrF — so changes to these scripts should preserve the
no-dependency constraint rather than reaching for a package.

## Commands

Run from the repo root.

```bash
# ASR transcription + WER/CER/RTF over audio/test{1,2,3}.wav
julia asr_pipeline.jl whisper.cpp/models/ggml-base.bin
# (must run after asr_pipeline.jl, since it reads out/*.txt)

# Translate out/*.txt via an Ollama model and score chrF
julia translate_pipeline.jl <ollama_model> <target_lang>
julia translate_pipeline.jl gemma4:12b-mlx English
```

- `asr_pipeline.jl` takes exactly one arg: a path to a whisper.cpp ggml model file (e.g. under
  `whisper.cpp/models/`). Language is hardcoded to `"es"` (`LANGUAGE` const).
- `translate_pipeline.jl` takes exactly two args: the Ollama model name/tag and the target
  language. It POSTs to `OLLAMA_URL` (`http://geoespacial.ucm.cl:11434/api/generate`, hardcoded
  const) — that host must be reachable for this script to work, and `curl` must be on `PATH`.
- Neither script has a test suite; validate changes by rerunning against the 3-file corpus in
  `audio/` and eyeballing the WER/CER/chrF table printed to stdout.

### whisper.cpp

The `whisper-cli` binary is prebuilt at `whisper.cpp/build/bin/whisper-cli`. Only rebuild it if
you've changed something under `whisper.cpp/` — that subtree is a large, independently maintained
upstream project; see `whisper.cpp/README.md` and `whisper.cpp/AGENTS.md` if you need to touch it
directly. Ggml models already present under `whisper.cpp/models/` include `ggml-base.bin` and
`ggml-tiny.bin`; other sizes can be fetched with `whisper.cpp/models/download-ggml-model.sh`.

## Reproducibility requirements (from the assignment rubric)

Per `estudio_casos_unidad1-2.tex`, any reported result must be reproducible and state: hardware
(CPU/GPU, OS), the `whisper.cpp` commit hash in use, the Ollama version, exact model name/tag
(including quantization, e.g. `Q8_0`/`Q5_1`/`Q4_0`), thread count, seeds, and decoding parameters.
Keep this in mind when adding scripts or output artifacts — capture/print this metadata rather
than only the metric values.
