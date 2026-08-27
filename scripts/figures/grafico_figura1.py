import os

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")


# ============================================================
# DATOS OFICIALES
# ============================================================

formatos = ["F32", "F16", "Q8_0", "Q5_1", "Q4_0"]

bits = np.array([
    32.0,
    16.0,
    8.5,
    6.0,
    4.5
])

wer = np.array([
    9.9502488,
    9.9502488,
    11.4427861,
    12.6865672,
    12.9353234
])

chrf = np.array([
    80.8978386,
    80.8978386,
    80.6041914,
    80.4368706,
    78.6713002
])


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})


# Usamos posiciones categóricas para que los modelos
# queden espaciados de forma uniforme y legible.
x = np.arange(len(formatos))

fig, axes = plt.subplots(
    1,
    2,
    figsize=(8.2, 3.6)
)

ax1, ax2 = axes


# ============================================================
# PANEL A — WER
# ============================================================

ax1.plot(
    x,
    wer,
    marker="o",
    linewidth=1.8,
    markersize=6
)

ax1.set_title(
    "(a) Error de transcripción"
)

ax1.set_ylabel(
    "WER de corpus (%)"
)

ax1.set_xticks(x)

ax1.set_xticklabels([
    "F32\n32 bits",
    "F16\n16 bits",
    "Q8_0\n8,5 bits",
    "Q5_1\n6 bits",
    "Q4_0\n4,5 bits"
])

ax1.set_ylim(
    9.4,
    13.5
)

ax1.grid(
    axis="y",
    alpha=0.25
)

# Valores numéricos
for xi, yi in zip(x, wer):

    ax1.annotate(
        f"{yi:.2f}",
        xy=(xi, yi),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5
    )


# ============================================================
# PANEL B — chrF
# ============================================================

ax2.plot(
    x,
    chrf,
    marker="o",
    linewidth=1.8,
    markersize=6
)

ax2.set_title(
    "(b) Calidad de traducción"
)

ax2.set_ylabel(
    "chrF de corpus"
)

ax2.set_xticks(x)

ax2.set_xticklabels([
    "F32\n32 bits",
    "F16\n16 bits",
    "Q8_0\n8,5 bits",
    "Q5_1\n6 bits",
    "Q4_0\n4,5 bits"
])

ax2.set_ylim(
    78.0,
    81.5
)

ax2.grid(
    axis="y",
    alpha=0.25
)

for xi, yi in zip(x, chrf):

    # En Q4 ponemos el valor arriba igualmente,
    # pero el rango deja suficiente espacio.
    ax2.annotate(
        f"{yi:.2f}",
        xy=(xi, yi),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5
    )


# ============================================================
# LIMPIEZA VISUAL
# ============================================================

for ax in axes:

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="x",
        length=0
    )


# Título general pequeño
fig.suptitle(
    "Precisión de la cascada según la representación numérica",
    fontsize=12,
    y=1.02
)

fig.tight_layout(
    pad=1.4,
    w_pad=2.2
)


# ============================================================
# EXPORTACIÓN
# ============================================================

fig.savefig(
    os.path.join(FIGURES_DIR, "figura_1_precision_bits.svg"),
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(FIGURES_DIR, "figura_1_precision_bits.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()