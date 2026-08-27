#!/usr/bin/env python3

import csv
import hashlib
import os
import statistics
import subprocess
import time
import wave


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WHISPER_CLI = os.path.join(
    BASE_DIR,
    "whisper.cpp",
    "build",
    "bin",
    "whisper-cli",
)

MODELS = {
    "F32": os.path.join(
        BASE_DIR,
        "whisper.cpp",
        "models",
        "ggml-base-f32.bin",
    ),
    "Q4_0": os.path.join(
        BASE_DIR,
        "whisper.cpp",
        "models",
        "ggml-base-q4_0.bin",
    ),
}

AUDIOS = [
    "audio01(dolor toracico).wav",
    "audio03(accidente).wav",
    "audio07(respiracion).wav",
]

AUDIO_DIR = os.path.join(
    BASE_DIR,
    "data",
    "audio",
)

THREADS = [
    1,
    2,
    4,
    8,
]

REPETITIONS = 3

LANGUAGE = "en"

OUT_RAW = os.path.join(
    BASE_DIR,
    "results", "csv",
    "thread_experiment.csv",
)

OUT_SUMMARY = os.path.join(
    BASE_DIR,
    "results", "csv",
    "thread_summary.csv",
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "output",
    "threads",
)


# ============================================================
# UTILIDADES
# ============================================================

def audio_duration(path):

    with wave.open(path, "rb") as wav:

        frames = wav.getnframes()
        rate = wav.getframerate()

        return frames / float(rate)


def normalize_text(text):

    return " ".join(
        text.strip().lower().split()
    )


def text_hash(text):

    return hashlib.sha256(
        normalize_text(text).encode("utf-8")
    ).hexdigest()


# ============================================================
# WHISPER
# ============================================================

def run_whisper(
    model_path,
    audio_path,
    threads,
    output_prefix,
):

    cmd = [
        WHISPER_CLI,

        "-m",
        model_path,

        "-f",
        audio_path,

        "-l",
        LANGUAGE,

        "-t",
        str(threads),

        "-bs",
        "5",

        "-bo",
        "5",

        "-tp",
        "0",

        "-tpi",
        "0",

        "-nf",

        "-nt",
        "-np",

        "-otxt",
        "-of",
        output_prefix,
    ]

    start = time.perf_counter()

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    if result.returncode != 0:

        raise RuntimeError(
            "whisper-cli falló.\n"
            + " ".join(cmd)
        )

    txt_path = (
        output_prefix
        + ".txt"
    )

    if not os.path.isfile(txt_path):

        raise RuntimeError(
            f"No se generó {txt_path}"
        )

    with open(
        txt_path,
        "r",
        encoding="utf-8",
    ) as f:

        hypothesis = f.read().strip()

    return (
        hypothesis,
        elapsed,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("EXPERIMENTO DE SENSIBILIDAD AL NÚMERO DE HILOS")
    print("=" * 72)
    print()

    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    if not os.path.isfile(
        WHISPER_CLI
    ):

        raise FileNotFoundError(
            WHISPER_CLI
        )

    for model_name, path in MODELS.items():

        if not os.path.isfile(path):

            raise FileNotFoundError(
                path
            )

    for audio in AUDIOS:

        path = os.path.join(
            AUDIO_DIR,
            audio,
        )

        if not os.path.isfile(path):

            raise FileNotFoundError(
                path
            )

    rows = []

    total_cases = (
        len(MODELS)
        * len(AUDIOS)
        * len(THREADS)
        * REPETITIONS
    )

    current = 0

    # ========================================================
    # EJECUCIÓN
    # ========================================================

    for model_name, model_path in MODELS.items():

        for audio_name in AUDIOS:

            audio_path = os.path.join(
                AUDIO_DIR,
                audio_name,
            )

            audio_s = audio_duration(
                audio_path
            )

            base_audio_name = (
                os.path.splitext(
                    audio_name
                )[0]
            )

            for threads in THREADS:

                for repetition in range(
                    1,
                    REPETITIONS + 1,
                ):

                    current += 1

                    output_prefix = os.path.join(
                        OUT_DIR,
                        (
                            f"{model_name}_"
                            f"{base_audio_name}_"
                            f"t{threads}_"
                            f"r{repetition}"
                        ),
                    )

                    print(
                        f"[{current:02d}/{total_cases}] "
                        f"{model_name:5s} | "
                        f"{audio_name:35s} | "
                        f"threads={threads} | "
                        f"run={repetition}"
                    )

                    hypothesis, elapsed = run_whisper(
                        model_path,
                        audio_path,
                        threads,
                        output_prefix,
                    )

                    rtf = (
                        elapsed
                        / audio_s
                    )

                    rows.append({
                        "model":
                            model_name,

                        "audio":
                            audio_name,

                        "threads":
                            threads,

                        "repetition":
                            repetition,

                        "audio_seconds":
                            audio_s,

                        "inference_seconds":
                            elapsed,

                        "rtf":
                            rtf,

                        "hypothesis":
                            hypothesis,

                        "hypothesis_hash":
                            text_hash(
                                hypothesis
                            ),
                    })

                    print(
                        "     RTF = "
                        f"{rtf:.6f}"
                    )

    # ========================================================
    # BUSCAR TRANSCRIPCIÓN BASE
    #
    # La referencia experimental es threads=4, repetición 1.
    # ========================================================

    baseline = {}

    for row in rows:

        if (
            row["threads"] == 4
            and
            row["repetition"] == 1
        ):

            key = (
                row["model"],
                row["audio"],
            )

            baseline[key] = (
                row[
                    "hypothesis_hash"
                ]
            )

    for row in rows:

        key = (
            row["model"],
            row["audio"],
        )

        row[
            "same_as_threads4_baseline"
        ] = (
            row["hypothesis_hash"]
            ==
            baseline[key]
        )

    # ========================================================
    # GUARDAR DATOS CRUDOS
    # ========================================================

    raw_fields = [
        "model",
        "audio",
        "threads",
        "repetition",
        "audio_seconds",
        "inference_seconds",
        "rtf",
        "hypothesis",
        "hypothesis_hash",
        "same_as_threads4_baseline",
    ]

    with open(
        OUT_RAW,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=raw_fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # RESUMEN
    # ========================================================

    summary_rows = []

    for model_name in MODELS:

        for threads in THREADS:

            group = [
                row
                for row in rows
                if
                row["model"]
                == model_name
                and
                row["threads"]
                == threads
            ]

            rtfs = [
                row["rtf"]
                for row in group
            ]

            inference_times = [
                row[
                    "inference_seconds"
                ]
                for row in group
            ]

            different = sum(
                not row[
                    "same_as_threads4_baseline"
                ]
                for row in group
            )

            unique_hypotheses = len(
                set(
                    row[
                        "hypothesis_hash"
                    ]
                    for row in group
                )
            )

            summary_rows.append({
                "model":
                    model_name,

                "threads":
                    threads,

                "cases":
                    len(group),

                "rtf_mean":
                    statistics.mean(
                        rtfs
                    ),

                "rtf_sd":
                    (
                        statistics.stdev(
                            rtfs
                        )
                        if len(rtfs) > 1
                        else 0.0
                    ),

                "inference_seconds_mean":
                    statistics.mean(
                        inference_times
                    ),

                "transcripts_different_from_threads4":
                    different,

                "unique_hypotheses":
                    unique_hypotheses,
            })

    summary_fields = [
        "model",
        "threads",
        "cases",
        "rtf_mean",
        "rtf_sd",
        "inference_seconds_mean",
        "transcripts_different_from_threads4",
        "unique_hypotheses",
    ]

    with open(
        OUT_SUMMARY,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields,
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    # ========================================================
    # RESULTADOS
    # ========================================================

    print()
    print("=" * 72)
    print("RESUMEN")
    print("=" * 72)
    print()

    for row in summary_rows:

        print(
            f"{row['model']:5s} | "
            f"threads={row['threads']:2d} | "
            f"RTF={row['rtf_mean']:.6f} "
            f"± {row['rtf_sd']:.6f} | "
            f"cambios={row['transcripts_different_from_threads4']} | "
            f"hipótesis únicas={row['unique_hypotheses']}"
        )

    print()
    print("Archivos creados:")
    print(
        " ",
        OUT_RAW,
    )
    print(
        " ",
        OUT_SUMMARY,
    )
    print()


if __name__ == "__main__":
    main()