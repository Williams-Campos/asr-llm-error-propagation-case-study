# Estudio de Caso: Cuantización y Propagación del Error en Traducción en Tiempo Real

## Información General

| Campo | Detalle |
|---|---|
| Actividad curricular | Computación Numérica (INF-321) |
| Carrera | Ingeniería Civil Informática |
| Académico | Sergio Hernández |
| Ponderación | 5 % |
| Fecha de entrega | 26 de agosto |
| Puntaje máximo | 80 puntos |
| Puntaje de corte (nota 4,0) | 48 puntos (60 % de exigencia) |
| Modalidad | Grupal (2 a 3 integrantes) e individual |

## Resultados de Aprendizaje Evaluados

Comprender la naturaleza de los errores numéricos en dispositivos de cálculo, considerando la investigación como estrategia de identificación de problemas.

## Indicadores de Evaluación

- Aplica correctamente la aritmética de punto flotante.
- Determina errores de propagación en operaciones.
- Maneja dispositivos de cálculo, entendiendo la representación y operatoria de los números.
- Utiliza software de apoyo.
- Utiliza la investigación en la determinación de problemas de aplicación.

## Instrucciones para el Desarrollo de la Evaluación

1. El trabajo debe presentarse en forma grupal e individual.
2. **Reproducibilidad total**: todo experimento debe ser reproducible. Se debe indicar hardware (CPU/GPU, sistema operativo), versión de whisper.cpp (commit hash), versión de Ollama, nombre y tag exacto de cada modelo, número de hilos, semillas y parámetros de decodificación utilizados. **Los resultados que no puedan reproducirse no reciben puntaje.**
3. Todo valor numérico reportado debe indicar el número de cifras significativas empleadas y, cuando corresponda, si se trata de un error absoluto o relativo. No se aceptan cifras sin unidad ni sin definición del error utilizado.
4. Etiquetar y titular figuras y tablas (ej. "Figura 1", "Figura 1.3" para la tercera figura de la sección 1). Referirse a ellas por su nombre en el texto, nunca como "la figura de arriba".
5. Las figuras deben estar centradas en la página, con título centrado debajo en cursiva. Deben tener un tamaño adecuado: ni tan pequeñas que se pierdan los detalles, ni más grandes de lo necesario.
6. El código o la salida de terminal (transcripción de shell) debe formatearse con fuente monoespaciada (ej. Consolas, DejaVu Sans Mono, Menlo).
7. **No se permite** incluir código, gráficos, tablas ni transcripciones de shell como captura de pantalla (imagen rasterizada). Esto se penaliza porque produce baja resolución, no se puede copiar/resaltar, y es inaccesible para lectores de pantalla.

### Restricciones

- Se aplicará el **artículo 67º del reglamento del estudiante**: la copia parcial o exacta (entre compañeros o de otra fuente) implica nota **1,0 para todos los involucrados**.
- Si un requerimiento queda sin contestar y no puede observarse en la rúbrica, **no aplica puntaje** para ese ítem.
- **Los experimentos deben ejecutarse localmente (on-device)**: no se acepta el uso de servicios de transcripción o traducción en la nube (APIs comerciales), ya que el objetivo es controlar la representación numérica de los modelos.
- El corpus de audio debe ser de libre uso o grabación propia con consentimiento de los hablantes; se debe adjuntar la referencia de la fuente.

## Descripción del Caso

Un centro de atención de urgencia de la Región del Maule desea implementar un sistema de traducción en tiempo real para atender pacientes que no hablan español (ej. creole haitiano o inglés). Por confidencialidad de los datos clínicos, el sistema debe ejecutarse **completamente en el dispositivo local** (un computador de escritorio sin GPU dedicada), con **latencia inferior al tiempo real** (real-time factor < 1).

La arquitectura propuesta es una **cascada de dos etapas**:

1. **Reconocimiento automático del habla (ASR)** con [whisper.cpp](https://github.com/ggml-org/whisper.cpp), implementación en C/C++ del modelo Whisper sobre la biblioteca ggml.
2. **Traducción y normalización del texto** con un modelo de lenguaje local servido mediante [Ollama](https://ollama.com).

El equipo disponible no tiene memoria suficiente para ejecutar ambos modelos en precisión completa, por lo que se recurre a la **cuantización**:

- Los pesos, originalmente en punto flotante IEEE-754 de 32 bits (**binary32**, con *p* = 24 bits de mantisa y unidad de redondeo *u* = 2⁻²⁴ ≈ 5,96 × 10⁻⁸), se convierten a formatos de menor ancho de palabra: **F16, Q8_0, Q5_1, Q4_0, Q4_K_M**, entre otros.
- En los esquemas ggml/gguf, los pesos se agrupan en bloques de 32 valores que comparten un **factor de escala *d*** almacenado en punto flotante de 16 bits, de modo que cada peso se reconstruye como:

  **w̃ᵢ = d·qᵢ + m**, con qᵢ ∈ ℤ ∩ [q_mín, q_máx]

- Esto introduce un **error de representación**:

  **εᵢ = w̃ᵢ − wᵢ**

  acotado por medio paso de cuantización: **|εᵢ| ≤ d/2**

> Una versión preliminar del caso, con instrucciones de instalación y ejecución de los modelos, está disponible en: [github.com/shernandez-ucm/estudio_casos_asr](https://github.com/shernandez-ucm/estudio_casos_asr)

## Pregunta Central del Informe

El centro de salud solicita un informe técnico que responda:

> **¿Cuánta precisión numérica se puede sacrificar antes de que la traducción deje de ser clínicamente confiable?**

## Métricas de Evaluación

Para cuantificar (no solo describir) el efecto de los errores numéricos sobre la calidad del resultado final, se emplean tres métricas estándar:

- **WER (Word Error Rate)** y **CER (Character Error Rate)**: cuantifican la distancia de edición (inserciones, sustituciones, eliminaciones) entre la transcripción generada por whisper.cpp y la transcripción de referencia, normalizada por la longitud de esta última.
- **BLEU (Bilingual Evaluation Understudy)**: evalúa la calidad de la traducción producida por el modelo servido en Ollama, mediante coincidencia de n-gramas con una o más traducciones de referencia.

Estas métricas permiten trazar, para cada formato de cuantización, una **curva de degradación de la calidad**, identificando el punto en que el error numérico deja de ser despreciable desde el punto de vista clínico.

## Requerimiento General

Cada grupo (2 a 3 integrantes) debe **diseñar y ejecutar un experimento numérico** que cuantifique el efecto de la cuantización y la propagación del error en la cascada ASR → LLM, y comunicar sus resultados en un informe y una presentación.

## Requerimientos Específicos

### R1. Representación numérica y aritmética de punto flotante

a) Compilar whisper.cpp y descargar un modelo (se sugiere *base* o *small*) en formato F32/F16. Generar al menos **tres versiones cuantizadas** con la herramienta `quantize` incluida en el repositorio (ej. Q8_0, Q5_1, Q4_0).

b) Para cada formato, reportar:
   - Tamaño en disco
   - Memoria RAM utilizada
   - Bits efectivos por peso (incluyendo el costo amortizado del factor de escala por bloque)
   - Paso de cuantización *d*

### R2. Efecto de la cuantización sobre la precisión de la transcripción y la traducción

a) Construir un corpus de **al menos 10 segmentos de audio** (15–30 s) con transcripción y traducción de referencia (grabación propia o corpus abierto tipo Common Voice o LibriSpeech).

b) Ejecutar la transcripción y traducción con cada formato del modelo, midiendo:
   - WER y CER para la transcripción
   - BLEU o chrF para la traducción
   - Real-time factor (RTF) y consumo de memoria

c) Graficar la curva de **precisión versus bits por peso**, identificando el punto de quiebre a partir del cual el error crece de manera no lineal. Discutir el compromiso precisión–memoria–latencia usando lenguaje de análisis de error (error absoluto y relativo, error de truncamiento frente a error de redondeo).

### R3. Propagación del error en la cascada con Ollama

a) Servir un mismo modelo de lenguaje en Ollama con **al menos dos cuantizaciones distintas** (ej. q4_0 y q8_0 de un modelo de 7–8B) y usarlo para traducir, fijando **temperature = 0** para aislar el efecto numérico del efecto del muestreo.

b) Separar las fuentes de error mediante **tres condiciones experimentales**, reportando los tres errores en una misma tabla:
   1. Traducir la transcripción de referencia (error del LLM aislado)
   2. Traducir la salida del ASR en alta precisión
   3. Traducir la salida del ASR cuantizado (error compuesto)

### R4. Investigación e identificación del problema

a) Formular una **recomendación técnica** al centro de salud: qué formato usar, con qué margen de error esperado, y qué riesgos residuales existen para un uso clínico.

## Formato de Entrega

La entrega consta de:
- Un **informe breve** (máximo 6 páginas, sin contar anexos)
- Una **presentación grupal de 10 minutos**

### Estructura del informe/presentación

| Sección | Modalidad |
|---|---|
| Introducción y planteamiento del problema numérico | Grupal |
| Marco teórico: representación en punto flotante, cuantización y propagación del error | Grupal |
| Diseño experimental, software utilizado y resultados en tablas y figuras | Grupal |
| Análisis de la propagación del error en la cascada y conclusiones | Grupal |
| Revisión crítica de un artículo o documentación técnica | Individual |

Una vez finalizada la presentación grupal, **cada integrante** debe presentar su artículo (**1 diapositiva**) demostrando capacidad de análisis, síntesis, evaluación y aplicación del conocimiento.

**Tiempo total disponible por grupo: 20 minutos** (presentación grupal + individual).

### Material adicional a entregar

Se debe adjuntar un repositorio o carpeta comprimida con:
- Los scripts utilizados (Julia, Python u Octave)
- Los archivos de audio o su referencia
- Las salidas crudas de cada corrida

### Plantillas de tablas sugeridas

**Tabla de resultados por formato de cuantización:**

| Formato | Bits/peso | Memoria (MB) | WER (%) | chrF | RTF |
|---|---|---|---|---|---|
| F32 | 32 | | | | |
| F16 | 16 | | | | |
| Q8_0 | | | | | |
| Q5_1 | | | | | |
| Q4_0 | | | | | |

**Tabla de propagación del error en la cascada:**

| Condición experimental | Error de entrada | Error de salida | κ empírico |
|---|---|---|---|
| Referencia → LLM | 0 | | |
| ASR alta precisión → LLM | | | |
| ASR cuantizado → LLM | | | |

## Rúbrica de Evaluación

| Indicador | Destacado (10 pts) | Competente (7 pts) | Básico (5 pts) | En Desarrollo (2 pts) |
|---|---|---|---|---|
| **Presentación y formato** (Grupal) | Informe y presentación impecables, bien organizados, dentro del tiempo, con tablas y figuras rotuladas y sin errores tipográficos. | Buena presentación, mayormente organizada, con pocos errores tipográficos o de rotulación. | Presentación aceptable, pero con errores tipográficos o de organización. | Presentación desorganizada o con numerosos errores. |
| **Representación numérica y punto flotante** (Grupal) | Deduce correctamente las cotas de error de cada formato, las verifica experimentalmente y las contrasta con la unidad de redondeo de binary32/binary16. | Deduce y verifica las cotas de error de los formatos, con contraste parcial. | Describe los formatos y reporta tamaños, pero sin deducir ni verificar cotas de error. | No distingue los formatos ni aplica la aritmética de punto flotante. |
| **Diseño experimental y uso de software** (Grupal) | Experimento reproducible y bien controlado; usa whisper.cpp y Ollama con métricas apropiadas, corpus adecuado y scripts documentados. | Experimento correcto y reproducible, con métricas apropiadas pero controles o documentación parciales. | Ejecuta las herramientas y reporta resultados, sin control de variables ni reproducibilidad. | No logra ejecutar los experimentos o los resultados no son verificables. |
| **Manejo de dispositivos de cálculo** (Grupal) | Reporta memoria, real-time factor y bits por peso de cada formato, y evidencia el efecto del número de hilos y del orden de las reducciones sobre la salida. | Reporta memoria y latencia por formato y verifica la variabilidad al cambiar hilos o semilla. | Reporta memoria o latencia sin analizar su relación con la representación numérica. | No caracteriza el dispositivo ni el costo de cada formato. |
| **Análisis de propagación del error** (Grupal) | Separa las fuentes de error, estima el factor de amplificación, lo interpreta en términos de condicionamiento y contrasta con las cotas teóricas. | Separa las fuentes de error y estima la amplificación, con interpretación parcial. | Compara errores entre formatos sin descomponer ni cuantificar la propagación. | Análisis del error ausente o incorrecto. |
| **Investigación y evaluación crítica** (Individual) | Selecciona una fuente pertinente, evalúa críticamente sus fortalezas y debilidades y la vincula explícitamente con los resultados obtenidos. | Evalúa críticamente la fuente y la relaciona con el caso. | Resume la fuente e identifica alguna fortaleza o debilidad. | Revisión ausente, superficial o sin relación con el caso. |
| **Claridad y coherencia** (Individual) | Presenta las ideas de manera clara, concisa y lógica, con lenguaje técnico apropiado y uso correcto de cifras significativas. | Ideas claras y bien organizadas, con lenguaje técnico apropiado. | Utiliza un lenguaje apropiado, con exposición poco estructurada. | Exposición confusa o insuficiente. |
| **Conclusiones y recomendación técnica** (Individual) | Conclusión fundamentada en la evidencia numérica, con recomendación explícita, margen de error esperado y riesgos residuales del uso clínico. | Conclusión clara y bien relacionada con los objetivos, con recomendación fundamentada. | Conclusión presente pero poco clara o insuficientemente relacionada con los resultados. | Conclusión ausente o irrelevante. |

**Puntaje total: máximo 80 puntos**

## Escala de Conversión Puntaje → Nota

Puntaje máximo: 80 puntos. Puntaje de corte (nota 4,0): 48 puntos, correspondiente a un 60 % de exigencia.

| Puntaje | Nota | Puntaje | Nota | Puntaje | Nota | Puntaje | Nota |
|---|---|---|---|---|---|---|---|
| 0 | 1,0 | 21 | 2,3 | 42 | 3,6 | 63 | 5,4 |
| 1 | 1,1 | 22 | 2,4 | 43 | 3,7 | 64 | 5,5 |
| 2 | 1,1 | 23 | 2,4 | 44 | 3,8 | 65 | 5,6 |
| 3 | 1,2 | 24 | 2,5 | 45 | 3,8 | 66 | 5,7 |
| 4 | 1,3 | 25 | 2,6 | 46 | 3,9 | 67 | 5,8 |
| 5 | 1,3 | 26 | 2,6 | 47 | 3,9 | 68 | 5,9 |
| 6 | 1,4 | 27 | 2,7 | 48 | 4,0 | 69 | 6,0 |
| 7 | 1,4 | 28 | 2,8 | 49 | 4,1 | 70 | 6,1 |
| 8 | 1,5 | 29 | 2,8 | 50 | 4,2 | 71 | 6,2 |
| 9 | 1,6 | 30 | 2,9 | 51 | 4,3 | 72 | 6,3 |
| 10 | 1,6 | 31 | 2,9 | 52 | 4,4 | 73 | 6,3 |
| 11 | 1,7 | 32 | 3,0 | 53 | 4,5 | 74 | 6,4 |
| 12 | 1,8 | 33 | 3,1 | 54 | 4,6 | 75 | 6,5 |
| 13 | 1,8 | 34 | 3,1 | 55 | 4,7 | 76 | 6,6 |
| 14 | 1,9 | 35 | 3,2 | 56 | 4,8 | 77 | 6,7 |
| 15 | 1,9 | 36 | 3,3 | 57 | 4,8 | 78 | 6,8 |
| 16 | 2,0 | 37 | 3,3 | 58 | 4,9 | 79 | 6,9 |
| 17 | 2,1 | 38 | 3,4 | 59 | 5,0 | 80 | 7,0 |
| 18 | 2,1 | 39 | 3,4 | 60 | 5,1 | | |
| 19 | 2,2 | 40 | 3,5 | 61 | 5,2 | | |
| 20 | 2,3 | 41 | 3,6 | 62 | 5,3 | | |