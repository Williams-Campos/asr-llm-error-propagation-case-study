#!/usr/bin/env python3

import csv
import math
import os
import struct
from dataclasses import dataclass

import numpy as np


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "whisper.cpp", "models")

MODELS = {
    "F32": os.path.join(MODEL_DIR, "ggml-base-f32.bin"),
    "F16": os.path.join(MODEL_DIR, "ggml-base-f16.bin"),
    "Q8_0": os.path.join(MODEL_DIR, "ggml-base-q8_0.bin"),
    "Q5_1": os.path.join(MODEL_DIR, "ggml-base-q5_1.bin"),
    "Q4_0": os.path.join(MODEL_DIR, "ggml-base-q4_0.bin"),
}

OUT_SUMMARY = os.path.join(
    BASE_DIR,
    "results", "csv",
    "quantization_error_summary.csv"
)

OUT_TENSOR = os.path.join(
    BASE_DIR,
    "results", "csv",
    "quantization_error_by_tensor.csv"
)


# ============================================================
# GGML TYPES
# ============================================================

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8

TYPE_NAMES = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_Q4_0: "Q4_0",
    GGML_TYPE_Q5_1: "Q5_1",
    GGML_TYPE_Q8_0: "Q8_0",
}

BLOCK_SIZE = {
    GGML_TYPE_F32: 1,
    GGML_TYPE_F16: 1,
    GGML_TYPE_Q4_0: 32,
    GGML_TYPE_Q5_1: 32,
    GGML_TYPE_Q8_0: 32,
}

TYPE_SIZE_BYTES = {
    GGML_TYPE_F32: 4,
    GGML_TYPE_F16: 2,

    # Q4_0:
    # FP16 d = 2 bytes
    # qs[16] = 16 bytes
    GGML_TYPE_Q4_0: 18,

    # Q5_1:
    # FP16 d = 2
    # FP16 m = 2
    # qh[4]  = 4
    # qs[16] = 16
    GGML_TYPE_Q5_1: 24,

    # Q8_0:
    # FP16 d = 2
    # int8 qs[32] = 32
    GGML_TYPE_Q8_0: 34,
}


# ============================================================
# ESTRUCTURAS
# ============================================================

@dataclass
class TensorInfo:
    name: str
    shape: tuple
    nelements: int
    type_id: int
    data_offset: int
    data_size: int


# ============================================================
# UTILIDADES BINARIAS
# ============================================================

def read_i32(f):
    raw = f.read(4)

    if len(raw) != 4:
        raise EOFError

    return struct.unpack("<i", raw)[0]


def fp16_from_bytes(raw):
    return np.frombuffer(
        raw,
        dtype="<f2"
    ).astype(np.float32)


def tensor_nbytes(type_id, nelements):

    if type_id not in BLOCK_SIZE:
        raise ValueError(
            f"Tipo GGML no soportado: {type_id}"
        )

    block = BLOCK_SIZE[type_id]
    type_size = TYPE_SIZE_BYTES[type_id]

    if nelements % block != 0:
        raise ValueError(
            f"{nelements} elementos no divisible "
            f"por block size {block}"
        )

    return (
        nelements // block
    ) * type_size


# ============================================================
# PARSER DEL FORMATO WHISPER GGML
# ============================================================

def parse_model(path):

    tensors = {}

    with open(path, "rb") as f:

        # ----------------------------------------------------
        # MAGIC
        # ----------------------------------------------------

        magic = f.read(4)

        if len(magic) != 4:
            raise ValueError(
                f"Archivo demasiado corto: {path}"
            )

        # El modelo legacy de Whisper comienza con "ggml"
        # Algunas versiones de whisper.cpp escriben el magic en little-endian ("lmgg")
        if magic not in (b"ggml", b"lmgg"):
            raise ValueError(
                f"Magic inesperado en {path}: {magic!r}"
            )

        # ----------------------------------------------------
        # HPARAMS DE WHISPER
        #
        # int32:
        # n_vocab
        # n_audio_ctx
        # n_audio_state
        # n_audio_head
        # n_audio_layer
        # n_text_ctx
        # n_text_state
        # n_text_head
        # n_text_layer
        # n_mels
        # ftype
        # ----------------------------------------------------

        hparams = [
            read_i32(f)
            for _ in range(11)
        ]

        n_vocab = hparams[0]

        # ----------------------------------------------------
        # MEL FILTERS
        # ----------------------------------------------------

        n_mel = read_i32(f)
        n_fft = read_i32(f)

        mel_count = n_mel * n_fft

        f.seek(
            4 * mel_count,
            os.SEEK_CUR
        )

        # ----------------------------------------------------
        # VOCABULARIO
        # ----------------------------------------------------

        vocab_size_file = read_i32(f)

        # Normalmente coincide con n_vocab.
        # Usamos el valor almacenado explícitamente.
        for _ in range(vocab_size_file):

            length = read_i32(f)

            f.seek(
                length,
                os.SEEK_CUR
            )

        # ----------------------------------------------------
        # TENSORES
        #
        # Cada tensor:
        #
        # int32 n_dims
        # int32 name_length
        # int32 type
        # int32 dimensions[n_dims]
        # char name[name_length]
        # raw tensor data
        # ----------------------------------------------------

        while True:

            header = f.read(12)

            if len(header) == 0:
                break

            if len(header) != 12:
                raise ValueError(
                    f"Header de tensor incompleto en {path}"
                )

            n_dims, name_length, type_id = struct.unpack(
                "<iii",
                header
            )

            if n_dims < 1 or n_dims > 4:
                raise ValueError(
                    f"n_dims inválido: {n_dims} "
                    f"en offset {f.tell() - 12}"
                )

            shape = tuple(
                read_i32(f)
                for _ in range(n_dims)
            )

            name_raw = f.read(name_length)

            if len(name_raw) != name_length:
                raise ValueError(
                    "Nombre de tensor incompleto"
                )

            name = name_raw.decode(
                "utf-8"
            )

            nelements = math.prod(shape)

            data_size = tensor_nbytes(
                type_id,
                nelements
            )

            data_offset = f.tell()

            tensors[name] = TensorInfo(
                name=name,
                shape=shape,
                nelements=nelements,
                type_id=type_id,
                data_offset=data_offset,
                data_size=data_size,
            )

            f.seek(
                data_size,
                os.SEEK_CUR
            )

    return tensors


# ============================================================
# DECODIFICADORES
# ============================================================

def decode_f32(raw, n):

    x = np.frombuffer(
        raw,
        dtype="<f4",
        count=n
    )

    return x.astype(
        np.float32,
        copy=True
    )


def decode_f16(raw, n):

    x = np.frombuffer(
        raw,
        dtype="<f2",
        count=n
    )

    return x.astype(
        np.float32
    )


def decode_q8_0(raw, n):

    if n % 32 != 0:
        raise ValueError(
            "Q8_0 requiere múltiplos de 32"
        )

    nb = n // 32

    out = np.empty(
        n,
        dtype=np.float32
    )

    pos = 0
    out_pos = 0

    for _ in range(nb):

        d = np.frombuffer(
            raw[pos:pos + 2],
            dtype="<f2"
        )[0].astype(np.float32)

        pos += 2

        qs = np.frombuffer(
            raw[pos:pos + 32],
            dtype=np.int8
        ).astype(np.float32)

        pos += 32

        out[
            out_pos:out_pos + 32
        ] = d * qs

        out_pos += 32

    return out


def decode_q4_0(raw, n):

    if n % 32 != 0:
        raise ValueError(
            "Q4_0 requiere múltiplos de 32"
        )

    nb = n // 32

    out = np.empty(
        n,
        dtype=np.float32
    )

    pos = 0
    out_pos = 0

    for _ in range(nb):

        d = np.frombuffer(
            raw[pos:pos + 2],
            dtype="<f2"
        )[0].astype(np.float32)

        pos += 2

        qs = np.frombuffer(
            raw[pos:pos + 16],
            dtype=np.uint8
        )

        pos += 16

        low = (
            qs & 0x0F
        ).astype(np.int16) - 8

        high = (
            qs >> 4
        ).astype(np.int16) - 8

        out[
            out_pos:out_pos + 16
        ] = d * low.astype(np.float32)

        out[
            out_pos + 16:out_pos + 32
        ] = d * high.astype(np.float32)

        out_pos += 32

    return out


def decode_q5_1(raw, n):

    if n % 32 != 0:
        raise ValueError(
            "Q5_1 requiere múltiplos de 32"
        )

    nb = n // 32

    out = np.empty(
        n,
        dtype=np.float32
    )

    pos = 0
    out_pos = 0

    for _ in range(nb):

        # d
        d = np.frombuffer(
            raw[pos:pos + 2],
            dtype="<f2"
        )[0].astype(np.float32)

        pos += 2

        # m
        m = np.frombuffer(
            raw[pos:pos + 2],
            dtype="<f2"
        )[0].astype(np.float32)

        pos += 2

        # qh[4]
        qh_bytes = raw[
            pos:pos + 4
        ]

        qh = int.from_bytes(
            qh_bytes,
            byteorder="little",
            signed=False
        )

        pos += 4

        # qs[16]
        qs = np.frombuffer(
            raw[pos:pos + 16],
            dtype=np.uint8
        )

        pos += 16

        for j in range(16):

            # Peso j
            high0 = (
                (
                    qh >> j
                ) & 1
            ) << 4

            q0 = int(
                qs[j] & 0x0F
            ) | high0

            # Peso j + 16
            high1 = (
                (
                    qh >> (j + 16)
                ) & 1
            ) << 4

            q1 = int(
                qs[j] >> 4
            ) | high1

            out[
                out_pos + j
            ] = d * q0 + m

            out[
                out_pos + j + 16
            ] = d * q1 + m

        out_pos += 32

    return out


def decode_tensor(
    path,
    info
):

    with open(path, "rb") as f:

        f.seek(
            info.data_offset
        )

        raw = f.read(
            info.data_size
        )

    if len(raw) != info.data_size:

        raise ValueError(
            f"Lectura incompleta de {info.name}"
        )

    if info.type_id == GGML_TYPE_F32:

        return decode_f32(
            raw,
            info.nelements
        )

    if info.type_id == GGML_TYPE_F16:

        return decode_f16(
            raw,
            info.nelements
        )

    if info.type_id == GGML_TYPE_Q8_0:

        return decode_q8_0(
            raw,
            info.nelements
        )

    if info.type_id == GGML_TYPE_Q5_1:

        return decode_q5_1(
            raw,
            info.nelements
        )

    if info.type_id == GGML_TYPE_Q4_0:

        return decode_q4_0(
            raw,
            info.nelements
        )

    raise ValueError(
        f"Tipo no soportado: {info.type_id}"
    )


# ============================================================
# EXTRAER d POR PESO
#
# Necesario para comprobar |epsilon| <= d/2
# ============================================================

def scales_per_weight(
    path,
    info
):

    if info.type_id not in (
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q4_0,
    ):

        return None

    with open(path, "rb") as f:

        f.seek(
            info.data_offset
        )

        raw = f.read(
            info.data_size
        )

    n = info.nelements
    nb = n // 32

    scales = np.empty(
        n,
        dtype=np.float32
    )

    pos = 0
    out_pos = 0

    for _ in range(nb):

        d = np.frombuffer(
            raw[pos:pos + 2],
            dtype="<f2"
        )[0].astype(np.float32)

        # Para la cota interesa el ancho positivo.
        d_abs = abs(
            float(d)
        )

        scales[
            out_pos:out_pos + 32
        ] = d_abs

        out_pos += 32

        if info.type_id == GGML_TYPE_Q8_0:

            pos += 34

        elif info.type_id == GGML_TYPE_Q5_1:

            pos += 24

        elif info.type_id == GGML_TYPE_Q4_0:

            pos += 18

    return scales


# ============================================================
# MÉTRICAS
# ============================================================

def metrics(
    reference,
    reconstructed,
    scales=None,
):

    reference = reference.astype(
        np.float64
    )

    reconstructed = reconstructed.astype(
        np.float64
    )

    error = (
        reconstructed -
        reference
    )

    abs_error = np.abs(
        error
    )

    squared_error = (
        error ** 2
    )

    mae = float(
        np.mean(abs_error)
    )

    median_abs = float(
        np.median(abs_error)
    )

    max_abs = float(
        np.max(abs_error)
    )

    rmse = float(
        np.sqrt(
            np.mean(
                squared_error
            )
        )
    )

    # --------------------------------------------------------
    # ERROR RELATIVO
    #
    # Se excluyen pesos muy cercanos a cero para evitar
    # cocientes numéricamente sin sentido.
    # --------------------------------------------------------

    threshold = 1e-8

    mask = (
        np.abs(reference)
        > threshold
    )

    if np.any(mask):

        rel_error = (
            abs_error[mask] /
            np.abs(reference[mask])
        )

        rel_mean = float(
            np.mean(rel_error)
        )

        rel_median = float(
            np.median(rel_error)
        )

        rel_max = float(
            np.max(rel_error)
        )

    else:

        rel_mean = math.nan
        rel_median = math.nan
        rel_max = math.nan

    result = {
        "n": len(reference),
        "mae": mae,
        "median_abs_error": median_abs,
        "max_abs_error": max_abs,
        "rmse": rmse,
        "relative_error_mean": rel_mean,
        "relative_error_median": rel_median,
        "relative_error_max": rel_max,
        "relative_threshold": threshold,
    }

    # --------------------------------------------------------
    # VERIFICACIÓN DE COTA
    #
    # |epsilon_i| <= d_i / 2
    #
    # Debido a que d y/o m se almacenan también en FP16,
    # pueden aparecer pequeñas violaciones de la cota
    # ideal derivada para la rejilla exacta.
    # --------------------------------------------------------

    if scales is not None:

        half_step = (
            scales.astype(np.float64)
            / 2.0
        )

        valid = (
            half_step > 0
        )

        if np.any(valid):

            ratio = (
                abs_error[valid] /
                half_step[valid]
            )

            # Tolerancia numérica muy pequeña para comparar
            # floats reconstruidos.
            tolerance = 1e-6

            passes = (
                ratio
                <= 1.0 + tolerance
            )

            result[
                "bound_valid_weights"
            ] = int(
                np.sum(valid)
            )

            result[
                "bound_pass_count"
            ] = int(
                np.sum(passes)
            )

            result[
                "bound_pass_percent"
            ] = (
                100.0
                * np.mean(passes)
            )

            result[
                "bound_ratio_mean"
            ] = float(
                np.mean(ratio)
            )

            result[
                "bound_ratio_median"
            ] = float(
                np.median(ratio)
            )

            result[
                "bound_ratio_max"
            ] = float(
                np.max(ratio)
            )

        else:

            result[
                "bound_valid_weights"
            ] = 0

            result[
                "bound_pass_count"
            ] = 0

            result[
                "bound_pass_percent"
            ] = math.nan

            result[
                "bound_ratio_mean"
            ] = math.nan

            result[
                "bound_ratio_median"
            ] = math.nan

            result[
                "bound_ratio_max"
            ] = math.nan

    else:

        result[
            "bound_valid_weights"
        ] = 0

        result[
            "bound_pass_count"
        ] = 0

        result[
            "bound_pass_percent"
        ] = math.nan

        result[
            "bound_ratio_mean"
        ] = math.nan

        result[
            "bound_ratio_median"
        ] = math.nan

        result[
            "bound_ratio_max"
        ] = math.nan

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("VERIFICACIÓN EXPERIMENTAL DEL ERROR DE REPRESENTACIÓN")
    print("=" * 72)
    print()

    # --------------------------------------------------------
    # Validar archivos
    # --------------------------------------------------------

    for name, path in MODELS.items():

        if not os.path.isfile(path):

            raise FileNotFoundError(
                f"No existe {name}: {path}"
            )

        print(
            f"{name:5s}: "
            f"{os.path.getsize(path):,} bytes"
        )

    print()
    print("Leyendo tablas GGML...")
    print()

    parsed = {}

    for name, path in MODELS.items():

        print(
            f"  -> {name}"
        )

        parsed[name] = parse_model(
            path
        )

        print(
            f"     tensores: "
            f"{len(parsed[name])}"
        )

    print()

    # --------------------------------------------------------
    # Modelo F32 como referencia
    # --------------------------------------------------------

    f32_tensors = parsed["F32"]

    tensor_rows = []

    global_data = {
        "F16": {
            "reference": [],
            "reconstructed": [],
            "scales": [],
        },
        "Q8_0": {
            "reference": [],
            "reconstructed": [],
            "scales": [],
        },
        "Q5_1": {
            "reference": [],
            "reconstructed": [],
            "scales": [],
        },
        "Q4_0": {
            "reference": [],
            "reconstructed": [],
            "scales": [],
        },
    }

    # --------------------------------------------------------
    # Comparar
    # --------------------------------------------------------

    for model_name in [
        "F16",
        "Q8_0",
        "Q5_1",
        "Q4_0",
    ]:

        print()
        print(
            "=" * 72
        )

        print(
            f"Comparando F32 vs {model_name}"
        )

        print(
            "=" * 72
        )

        target_tensors = parsed[
            model_name
        ]

        compared_tensors = 0
        compared_weights = 0

        for tensor_name, target_info in target_tensors.items():

            if tensor_name not in f32_tensors:
                continue

            ref_info = f32_tensors[
                tensor_name
            ]

            if (
                ref_info.shape
                != target_info.shape
            ):

                raise ValueError(
                    f"Shape distinta en {tensor_name}: "
                    f"{ref_info.shape} vs "
                    f"{target_info.shape}"
                )

            # ------------------------------------------------
            # Solo nos interesa medir un cambio de
            # representación.
            #
            # Si el tensor del modelo destino sigue siendo F32,
            # no forma parte del error producido por esa
            # reducción de precisión.
            # ------------------------------------------------

            if (
                target_info.type_id
                == GGML_TYPE_F32
            ):

                continue

            # Solo formatos que sabemos decodificar.
            if target_info.type_id not in (
                GGML_TYPE_F16,
                GGML_TYPE_Q8_0,
                GGML_TYPE_Q5_1,
                GGML_TYPE_Q4_0,
            ):

                print(
                    "SKIP tipo no soportado:",
                    tensor_name,
                    target_info.type_id,
                )

                continue

            # El tensor original debe poder interpretarse
            # exactamente como referencia numérica.
            ref_values = decode_tensor(
                MODELS["F32"],
                ref_info
            )

            target_values = decode_tensor(
                MODELS[model_name],
                target_info
            )

            if (
                len(ref_values)
                != len(target_values)
            ):

                raise ValueError(
                    f"Número de pesos distinto "
                    f"en {tensor_name}"
                )

            scales = scales_per_weight(
                MODELS[model_name],
                target_info
            )

            m = metrics(
                ref_values,
                target_values,
                scales
            )

            tensor_rows.append({
                "format":
                    model_name,

                "tensor":
                    tensor_name,

                "tensor_type":
                    TYPE_NAMES[
                        target_info.type_id
                    ],

                "shape":
                    "x".join(
                        map(
                            str,
                            target_info.shape
                        )
                    ),

                **m
            })

            global_data[
                model_name
            ][
                "reference"
            ].append(
                ref_values
            )

            global_data[
                model_name
            ][
                "reconstructed"
            ].append(
                target_values
            )

            if scales is not None:

                global_data[
                    model_name
                ][
                    "scales"
                ].append(
                    scales
                )

            else:

                global_data[
                    model_name
                ][
                    "scales"
                ].append(
                    np.full(
                        len(ref_values),
                        np.nan,
                        dtype=np.float32
                    )
                )

            compared_tensors += 1
            compared_weights += len(
                ref_values
            )

        print(
            f"Tensores comparados: "
            f"{compared_tensors}"
        )

        print(
            f"Pesos comparados: "
            f"{compared_weights:,}"
        )

    # ========================================================
    # RESUMEN GLOBAL
    # ========================================================

    summary_rows = []

    for model_name in [
        "F16",
        "Q8_0",
        "Q5_1",
        "Q4_0",
    ]:

        refs = global_data[
            model_name
        ][
            "reference"
        ]

        recs = global_data[
            model_name
        ][
            "reconstructed"
        ]

        if len(refs) == 0:
            continue

        ref_all = np.concatenate(
            refs
        )

        rec_all = np.concatenate(
            recs
        )

        scales_list = global_data[
            model_name
        ][
            "scales"
        ]

        if len(scales_list) > 0:

            scales_all = np.concatenate(
                scales_list
            )

        else:

            scales_all = None

        m = metrics(
            ref_all,
            rec_all,
            scales_all
        )

        summary_rows.append({
            "format":
                model_name,

            "tensors_compared":
                len(refs),

            **m,
        })

    # ========================================================
    # ESCRIBIR CSV POR TENSOR
    # ========================================================

    tensor_fields = [
        "format",
        "tensor",
        "tensor_type",
        "shape",
        "n",
        "mae",
        "median_abs_error",
        "max_abs_error",
        "rmse",
        "relative_error_mean",
        "relative_error_median",
        "relative_error_max",
        "relative_threshold",
        "bound_valid_weights",
        "bound_pass_count",
        "bound_pass_percent",
        "bound_ratio_mean",
        "bound_ratio_median",
        "bound_ratio_max",
    ]

    with open(
        OUT_TENSOR,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=tensor_fields
        )

        writer.writeheader()

        writer.writerows(
            tensor_rows
        )

    # ========================================================
    # ESCRIBIR RESUMEN
    # ========================================================

    summary_fields = [
        "format",
        "tensors_compared",
        "n",
        "mae",
        "median_abs_error",
        "max_abs_error",
        "rmse",
        "relative_error_mean",
        "relative_error_median",
        "relative_error_max",
        "relative_threshold",
        "bound_valid_weights",
        "bound_pass_count",
        "bound_pass_percent",
        "bound_ratio_mean",
        "bound_ratio_median",
        "bound_ratio_max",
    ]

    with open(
        OUT_SUMMARY,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    # ========================================================
    # MOSTRAR RESULTADOS
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "RESUMEN"
    )

    print(
        "=" * 72
    )

    print()

    for row in summary_rows:

        print(
            row["format"]
        )

        print(
            f"  pesos comparados: "
            f"{row['n']:,}"
        )

        print(
            f"  MAE:              "
            f"{row['mae']:.8e}"
        )

        print(
            f"  mediana |error|:  "
            f"{row['median_abs_error']:.8e}"
        )

        print(
            f"  max |error|:      "
            f"{row['max_abs_error']:.8e}"
        )

        print(
            f"  RMSE:             "
            f"{row['rmse']:.8e}"
        )

        print(
            f"  error rel mediana:"
            f" {row['relative_error_median']:.8e}"
        )

        if not math.isnan(
            row[
                "bound_pass_percent"
            ]
        ):

            print(
                f"  cota d/2 cumplida:"
                f" {row['bound_pass_percent']:.6f}%"
            )

            print(
                f"  max(|e|/(d/2)):   "
                f"{row['bound_ratio_max']:.8f}"
            )

        else:

            print(
                "  cota d/2:          "
                "N/A"
            )

        print()

    print(
        "Archivos creados:"
    )

    print(
        f"  {OUT_SUMMARY}"
    )

    print(
        f"  {OUT_TENSOR}"
    )

    print()


if __name__ == "__main__":
    main()