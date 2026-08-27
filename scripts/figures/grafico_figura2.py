import os

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")


# ============================================================
# DATOS OFICIALES
# ============================================================

formatos = [
    "F16",
    "Q8_0",
    "Q5_1",
    "Q4_0"
]

mae = np.array([
    0.0,
    0.0001181878657266,
    0.0007226811988958,
    0.0018697546757584
])

x = np.arange(len(formatos))


# ============================================================
# ESTILO GENERAL
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})


fig, ax = plt.subplots(
    figsize=(6.6, 3.7)
)


# ============================================================
# CURVA
# ============================================================

ax.plot(
    x,
    mae,
    marker="o",
    linewidth=2.0,
    markersize=6.5
)


# ============================================================
# EJE X
# ============================================================

ax.set_xticks(x)

ax.set_xticklabels([
    "F16\n16 bits",
    "Q8_0\n8,5 bits",
    "Q5_1\n6 bits",
    "Q4_0\n4,5 bits"
])

ax.tick_params(
    axis="x",
    length=0,
    pad=7
)


# ============================================================
# EJE Y
# ============================================================

ax.set_ylabel(
    r"MAE de los pesos ($\times 10^{-3}$)"
)

# Escalamos visualmente el eje a unidades de 10^-3
ax.set_yticks([
    0.0000,
    0.0005,
    0.0010,
    0.0015,
    0.0020
])

ax.set_yticklabels([
    "0",
    "0,5",
    "1,0",
    "1,5",
    "2,0"
])

ax.set_ylim(
    -0.00005,
    0.00208
)

# Más espacio horizontal para que Q4_0 no quede pegado al borde
ax.set_xlim(
    -0.15,
    3.18
)


# ============================================================
# VALORES SOBRE LOS PUNTOS
# ============================================================

etiquetas = [
    "0",
    "0,118",
    "0,723",
    "1,870"
]

offsets = [
    (0, 8),
    (0, 9),
    (0, 9),
    (0, 9)
]

for xi, yi, texto, offset in zip(
    x,
    mae,
    etiquetas,
    offsets
):

    ax.annotate(
        texto,
        xy=(xi, yi),
        xytext=offset,
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9
    )


# ============================================================
# LIMPIEZA VISUAL
# ============================================================

ax.grid(
    axis="y",
    alpha=0.22,
    linewidth=0.8
)

ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

ax.tick_params(
    axis="y",
    width=0.8
)


# ============================================================
# TÍTULO
# ============================================================

ax.set_title(
    "Error de representación de los pesos",
    fontsize=11,
    pad=10
)


# ============================================================
# AJUSTE FINAL
# ============================================================

fig.tight_layout(
    pad=1.1
)


# ============================================================
# EXPORTACIÓN
# ============================================================

fig.savefig(
    os.path.join(FIGURES_DIR, "figura_2_error_representacion.svg"),
    bbox_inches="tight"
)

fig.savefig(
    os.path.join(FIGURES_DIR, "figura_2_error_representacion.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()