import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# DATOS OFICIALES
# ============================================================

condiciones = [
    "Referencia",
    "F32",
    "Q8_0",
    "Q5_1",
    "Q4_0"
]

gemma_q8 = np.array([
    85.28284998238303,
    80.89783862707091,
    80.60419141738078,
    80.43687061016585,
    78.67130021302418
])

gemma_q4 = np.array([
    86.10602468709298,
    80.48163157099397,
    81.02340347390111,
    79.32483648250478,
    78.04355138540616
])


# ============================================================
# ESTILO GENERAL
# ============================================================

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})

x = np.arange(len(condiciones))

fig, ax = plt.subplots(
    figsize=(7.0, 3.9)
)


# ============================================================
# CURVAS
# ============================================================

ax.plot(
    x,
    gemma_q8,
    marker="o",
    linewidth=2.0,
    markersize=6.5,
    label="Gemma Q8_0"
)

ax.plot(
    x,
    gemma_q4,
    marker="o",
    linewidth=2.0,
    markersize=6.5,
    label="Gemma Q4_K_M"
)


# ============================================================
# EJE X
# ============================================================

ax.set_xticks(x)

ax.set_xticklabels([
    "Referencia",
    "F32",
    "Q8_0",
    "Q5_1",
    "Q4_0"
])

ax.set_xlabel(
    "Entrada al modelo de traducción"
)

ax.tick_params(
    axis="x",
    length=0,
    pad=7
)


# ============================================================
# EJE Y
# ============================================================

ax.set_ylabel(
    "chrF de corpus"
)

ax.set_ylim(
    77.2,
    87.0
)

ax.set_yticks([
    78,
    80,
    82,
    84,
    86
])


# ============================================================
# POSICIONES MANUALES DE ETIQUETAS
#
# (desplazamiento horizontal, desplazamiento vertical)
# en puntos gráficos.
# ============================================================

offsets_q8 = [
    (0, -17),   # Referencia: debajo del punto azul
    (0, 10),    # F32: arriba
    (0, -17),   # Q8: debajo
    (0, 10),    # Q5: arriba
    (0, 10)     # Q4: arriba
]

offsets_q4 = [
    (0, 10),    # Referencia: arriba del punto naranja
    (0, -17),   # F32: debajo
    (0, 10),    # Q8: arriba
    (0, -17),   # Q5: debajo
    (0, 11)     # Q4: ARRIBA para evitar el eje X
]


# Fondo pequeño detrás del número.
# Evita que una curva pase visualmente por encima del texto.
label_box = dict(
    boxstyle="round,pad=0.15",
    facecolor="white",
    edgecolor="none",
    alpha=0.85
)


# ============================================================
# ETIQUETAS GEMMA Q8
# ============================================================

for xi, yi, offset in zip(
    x,
    gemma_q8,
    offsets_q8
):

    ax.annotate(
        f"{yi:.2f}".replace(".", ","),
        xy=(xi, yi),
        xytext=offset,
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=8.2,
        bbox=label_box
    )


# ============================================================
# ETIQUETAS GEMMA Q4
# ============================================================

for xi, yi, offset in zip(
    x,
    gemma_q4,
    offsets_q4
):

    ax.annotate(
        f"{yi:.2f}".replace(".", ","),
        xy=(xi, yi),
        xytext=offset,
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=8.2,
        bbox=label_box
    )


# ============================================================
# LEYENDA
# ============================================================

ax.legend(
    frameon=False,
    loc="upper right"
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

ax.set_xlim(
    -0.15,
    4.20
)


# ============================================================
# TÍTULO
# ============================================================

ax.set_title(
    "Propagación del error en la cascada ASR → LLM",
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
    "figura_3_propagacion_asr_llm.svg",
    bbox_inches="tight"
)

fig.savefig(
    "figura_3_propagacion_asr_llm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()