#!/usr/bin/env python3

import csv
import math
import os
import struct

import numpy as np


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "whisper.cpp",
    "models",
    "ggml-base-f32.bin",
)

TARGET_TENSOR = (
    "encoder.blocks.0."
    "attn.query.weight"
)

OUT_FILE = os.path.join(
    BASE_DIR,
    "results", "csv",
    "reduction_order_experiment.csv",
)

SEED = 42


# ============================================================
# LECTOR GGML LEGACY WHISPER
# ============================================================

def read_i32(f):

    raw = f.read(4)

    if len(raw) != 4:
        raise EOFError

    return struct.unpack(
        "<i",
        raw
    )[0]


def find_tensor(
    path,
    target_name,
):

    with open(
        path,
        "rb",
    ) as f:

        magic = f.read(4)

        if magic not in (b"ggml", b"lmgg"):

            raise ValueError(
                f"Magic inesperado: {magic!r}"
            )

        # Whisper hparams
        for _ in range(11):
            read_i32(f)

        # Mel filters
        n_mel = read_i32(f)
        n_fft = read_i32(f)

        f.seek(
            n_mel
            * n_fft
            * 4,
            os.SEEK_CUR,
        )

        # Vocabulary
        vocab_size = read_i32(f)

        for _ in range(
            vocab_size
        ):

            length = read_i32(f)

            f.seek(
                length,
                os.SEEK_CUR,
            )

        # Tensors
        while True:

            header = f.read(12)

            if len(header) == 0:
                break

            if len(header) != 12:
                raise ValueError(
                    "Tensor header incompleto"
                )

            n_dims, name_len, type_id = (
                struct.unpack(
                    "<iii",
                    header,
                )
            )

            shape = tuple(
                read_i32(f)
                for _ in range(
                    n_dims
                )
            )

            name = (
                f.read(
                    name_len
                )
                .decode("utf-8")
            )

            n = math.prod(
                shape
            )

            if type_id == 0:

                data_size = (
                    n * 4
                )

            elif type_id == 1:

                data_size = (
                    n * 2
                )

            else:

                raise ValueError(
                    "Se encontró un tipo "
                    f"no esperado en F32: {type_id}"
                )

            if (
                name
                == target_name
            ):

                raw = f.read(
                    data_size
                )

                if type_id == 0:

                    values = np.frombuffer(
                        raw,
                        dtype="<f4",
                    ).astype(
                        np.float32,
                        copy=True,
                    )

                else:

                    values = np.frombuffer(
                        raw,
                        dtype="<f2",
                    ).astype(
                        np.float32
                    )

                return (
                    values,
                    shape,
                    type_id,
                )

            f.seek(
                data_size,
                os.SEEK_CUR,
            )

    raise ValueError(
        f"No se encontró tensor: "
        f"{target_name}"
    )


# ============================================================
# REDUCCIONES FLOAT32
# ============================================================

def sum_left_to_right(
    values
):

    acc = np.float32(0.0)

    for x in values:

        acc = np.float32(
            acc
            +
            np.float32(x)
        )

    return acc


def sum_right_to_left(
    values
):

    acc = np.float32(0.0)

    for x in values[::-1]:

        acc = np.float32(
            acc
            +
            np.float32(x)
        )

    return acc


def sum_pairwise(
    values
):

    current = np.asarray(
        values,
        dtype=np.float32,
    ).copy()

    while (
        len(current)
        > 1
    ):

        if (
            len(current)
            % 2
            == 1
        ):

            tail = current[-1:]

            current = current[:-1]

        else:

            tail = None

        current = np.float32(
            current[0::2]
            +
            current[1::2]
        )

        if tail is not None:

            current = np.concatenate(
                [
                    current,
                    tail,
                ]
            )

    return np.float32(
        current[0]
    )


def sum_small_to_large(
    values
):

    order = np.argsort(
        np.abs(values)
    )

    sorted_values = (
        values[order]
    )

    return sum_left_to_right(
        sorted_values
    )


def sum_large_to_small(
    values
):

    order = np.argsort(
        np.abs(values)
    )[::-1]

    sorted_values = (
        values[order]
    )

    return sum_left_to_right(
        sorted_values
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("EXPERIMENTO DE ORDEN DE REDUCCIÓN FLOAT32")
    print("=" * 72)
    print()

    if not os.path.isfile(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            MODEL_PATH
        )

    weights, shape, type_id = find_tensor(
        MODEL_PATH,
        TARGET_TENSOR,
    )

    print(
        "Tensor:",
        TARGET_TENSOR,
    )

    print(
        "Shape:",
        shape,
    )

    print(
        "Número de pesos:",
        len(weights),
    )

    print()

    # ========================================================
    # VECTOR DETERMINISTA x
    #
    # Simula una entrada de una operación tipo dot-product.
    # ========================================================

    rng = np.random.default_rng(
        SEED
    )

    x = rng.standard_normal(
        len(weights)
    ).astype(
        np.float32
    )

    products = np.float32(
        weights
        *
        x
    )

    # ========================================================
    # REFERENCIA
    #
    # Acumulación en Float64 de LOS MISMOS productos Float32.
    #
    # Así aislamos principalmente el error de reducción,
    # no el error del producto.
    # ========================================================

    reference = float(
        np.sum(
            products.astype(
                np.float64
            ),
            dtype=np.float64,
        )
    )

    methods = {
        "float64_reference":
            reference,

        "float32_left_to_right":
            float(
                sum_left_to_right(
                    products
                )
            ),

        "float32_right_to_left":
            float(
                sum_right_to_left(
                    products
                )
            ),

        "float32_pairwise":
            float(
                sum_pairwise(
                    products
                )
            ),

        "float32_small_to_large":
            float(
                sum_small_to_large(
                    products
                )
            ),

        "float32_large_to_small":
            float(
                sum_large_to_small(
                    products
                )
            ),
    }

    rows = []

    for method, result in methods.items():

        abs_error = abs(
            result
            -
            reference
        )

        if reference != 0:

            rel_error = (
                abs_error
                /
                abs(reference)
            )

        else:

            rel_error = math.nan

        rows.append({
            "tensor":
                TARGET_TENSOR,

            "n":
                len(products),

            "seed":
                SEED,

            "method":
                method,

            "result":
                result,

            "reference_float64":
                reference,

            "absolute_error":
                abs_error,

            "relative_error":
                rel_error,
        })

    # ========================================================
    # CSV
    # ========================================================

    fields = [
        "tensor",
        "n",
        "seed",
        "method",
        "result",
        "reference_float64",
        "absolute_error",
        "relative_error",
    ]

    with open(
        OUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # RESULTADOS
    # ========================================================

    print(
        f"Referencia Float64 = "
        f"{reference:.16e}"
    )

    print()

    for row in rows:

        print(
            f"{row['method']:28s} "
            f"resultado="
            f"{row['result']:.16e} | "
            f"Eabs="
            f"{row['absolute_error']:.8e} | "
            f"Erel="
            f"{row['relative_error']:.8e}"
        )

    print()
    print(
        "Archivo creado:"
    )
    print(
        " ",
        OUT_FILE,
    )
    print()


if __name__ == "__main__":
    main()