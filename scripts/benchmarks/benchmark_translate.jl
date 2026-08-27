#!/usr/bin/env julia

using Statistics
using Dates


# ============================================================
# BENCHMARK R2 — TRADUCCIÓN
#
# Objetivo:
#
# Para cada formato de Whisper:
#
#   F32
#   F16
#   Q8_0
#   Q5_1
#   Q4_0
#
# tomar la transcripción inglesa ya obtenida en
# asr_per_audio.csv, traducirla al español usando siempre
# el mismo LLM:
#
#   gemma3:1b-it-q8_0
#
# y comparar la traducción generada directamente contra:
#
#   groundtruth_es.txt
#
# mediante chrF.
#
#
# IMPORTANTE:
#
# Aquí NO traducimos la referencia inglesa con el LLM.
#
# La referencia española verdadera es groundtruth_es.txt.
#
# ============================================================


# ============================================================
# CONFIGURACIÓN
# ============================================================

const BASE_DIR = dirname(dirname(@__DIR__))

const ASR_CSV =
    joinpath(BASE_DIR, "results", "csv", "asr_per_audio.csv")

const GROUNDTRUTH_ES =
    joinpath(BASE_DIR, "data", "groundtruth_es.txt")


const OUT_PER_AUDIO =
    joinpath(BASE_DIR, "results", "csv", "translation_per_audio.csv")

const OUT_SUMMARY =
    joinpath(BASE_DIR, "results", "csv", "translation_summary.csv")

const OUT_METADATA =
    joinpath(BASE_DIR, "results", "csv", "translation_metadata.csv")


# ------------------------------------------------------------
# Ollama
# ------------------------------------------------------------

const OLLAMA_URL = "http://localhost:11434/api/generate"


# ------------------------------------------------------------
# LLM FIJO PARA R2
#
# La idea de R2 es estudiar el efecto de cambiar la precisión
# de Whisper manteniendo el traductor constante.
# ------------------------------------------------------------

const LLM_MODEL = "gemma3:1b-it-q8_0"

const LLM_QUANTIZATION = "Q8_0"


# ------------------------------------------------------------
# Parámetros del LLM
# ------------------------------------------------------------

const TEMPERATURE = 0.0

const SEED = 42

const TARGET_LANGUAGE = "Spanish"


# ------------------------------------------------------------
# Modelos Whisper
# ------------------------------------------------------------

const ASR_MODELS = [
    "F32",
    "F16",
    "Q8_0",
    "Q5_1",
    "Q4_0"
]


# ------------------------------------------------------------
# Repetición del ASR que utilizaremos
#
# Las repeticiones dieron las mismas transcripciones.
# Por eso basta utilizar una.
# ------------------------------------------------------------

const ASR_REPETITION = 1


# ============================================================
# ESTRUCTURAS
# ============================================================

struct ASRRow

    model::String

    repetition::Int

    audio::String

    reference_en::String

    hypothesis_en::String

end


struct TranslationRow

    asr_model::String

    audio::String

    chrf::Float64

    reference_en::String

    hypothesis_en::String

    reference_es::String

    hypothesis_es::String

    total_duration_s::Float64

    load_duration_s::Float64

    prompt_eval_duration_s::Float64

    eval_duration_s::Float64

    prompt_eval_count::Int

    eval_count::Int

end


# ============================================================
# CSV SIMPLE
#
# Sin dependencias externas.
# ============================================================

function parse_csv_line(line::AbstractString)

    fields = String[]

    buffer = IOBuffer()

    in_quotes = false

    i = firstindex(line)


    while i <= lastindex(line)

        c = line[i]


        if c == '"'

            next_i = nextind(line, i)


            # "" dentro de una celda CSV representa
            # una comilla literal.
            if in_quotes &&
               next_i <= lastindex(line) &&
               line[next_i] == '"'

                print(buffer, '"')

                i = next_i

            else

                in_quotes = !in_quotes

            end


        elseif c == ',' && !in_quotes

            push!(
                fields,
                String(take!(buffer))
            )

        else

            print(buffer, c)

        end


        i = nextind(line, i)

    end


    push!(
        fields,
        String(take!(buffer))
    )


    return fields

end


function csv_escape(x)

    s = string(x)


    if occursin(',', s) ||
       occursin('"', s) ||
       occursin('\n', s) ||
       occursin('\r', s)

        return "\"" *
               replace(s, "\"" => "\"\"") *
               "\""

    end


    return s

end


function write_csv_row(io, values)

    println(
        io,
        join(
            csv_escape.(values),
            ","
        )
    )

end


# ============================================================
# GROUNDTRUTH ESPAÑOL
#
# Formato esperado:
#
# audio01.wav,"Traducción española..."
#
# exactamente igual al formato de groundtruth.txt.
# ============================================================

function read_groundtruth(path::String)

    refs =
        Dict{String,String}()


    for line in eachline(path)

        isempty(strip(line)) &&
            continue


        m =
            match(
                r"^([^,]+),\"(.*)\"$",
                line
            )


        if m === nothing

            error(
                "Línea inválida en groundtruth_es.txt:\n$line"
            )

        end


        filename =
            String(
                m.captures[1]
            )


        text =
            String(
                m.captures[2]
            )


        if haskey(refs, filename)

            error(
                "Audio repetido en groundtruth_es.txt: $filename"
            )

        end


        refs[filename] =
            text

    end


    return refs

end


# ============================================================
# JSON
# ============================================================

function json_escape(s::AbstractString)

    io = IOBuffer()


    for c in s

        if c == '"'

            print(io, "\\\"")


        elseif c == '\\'

            print(io, "\\\\")


        elseif c == '\n'

            print(io, "\\n")


        elseif c == '\r'

            print(io, "\\r")


        elseif c == '\t'

            print(io, "\\t")


        else

            print(io, c)

        end

    end


    return String(
        take!(io)
    )

end


const JSON_ESCAPES =
    Dict(
        'n' => '\n',
        't' => '\t',
        'r' => '\r',
        '"' => '"',
        '\\' => '\\',
        '/' => '/'
    )


function json_extract_string(
    json::AbstractString,
    key::String
)

    marker = "\"$key\":\""


    range =
        findfirst(
            marker,
            json
        )


    range === nothing &&
        error(
            "Campo '$key' no encontrado en la respuesta de Ollama:\n$json"
        )


    i =
        nextind(
            json,
            last(range)
        )


    io =
        IOBuffer()


    while i <= lastindex(json)

        c =
            json[i]


        if c == '"'

            break


        elseif c == '\\'

            i =
                nextind(
                    json,
                    i
                )


            e =
                json[i]


            if e == 'u'

                indices =
                    Int[]


                j =
                    nextind(
                        json,
                        i
                    )


                for _ in 1:4

                    push!(
                        indices,
                        j
                    )

                    j =
                        nextind(
                            json,
                            j
                        )

                end


                hex =
                    String(
                        json[indices]
                    )


                print(
                    io,
                    Char(
                        parse(
                            UInt32,
                            hex;
                            base=16
                        )
                    )
                )


                i =
                    indices[end]

            else

                print(
                    io,
                    get(
                        JSON_ESCAPES,
                        e,
                        e
                    )
                )

            end


        else

            print(
                io,
                c
            )

        end


        i =
            nextind(
                json,
                i
            )

    end


    return String(
        take!(io)
    )

end


function json_extract_number(
    json::AbstractString,
    key::String;
    default=0
)

    pattern =
        Regex(
            "\"$(key)\"\\s*:\\s*([0-9]+(?:\\.[0-9]+)?)"
        )


    m =
        match(
            pattern,
            json
        )


    m === nothing &&
        return default


    value =
        m.captures[1]


    if occursin('.', value)

        return parse(
            Float64,
            value
        )

    else

        return parse(
            Int,
            value
        )

    end

end


# ============================================================
# NORMALIZACIÓN
# ============================================================

function normalize_text(
    text::AbstractString
)

    text =
        lowercase(
            strip(text)
        )


    text =
        replace(
            text,
            r"\s+" => " "
        )


    return text

end


# ============================================================
# chrF
#
# Character n-gram F-score
#
# max_n = 6
# beta  = 2
#
# Igual criterio que usamos anteriormente.
# ============================================================

function char_ngrams(
    text::AbstractString,
    n::Int
)

    chars =
        collect(text)


    counts =
        Dict{String,Int}()


    if length(chars) < n

        return counts

    end


    for i in 1:(length(chars)-n+1)

        gram =
            String(
                chars[
                    i:i+n-1
                ]
            )


        counts[gram] =
            get(
                counts,
                gram,
                0
            ) + 1

    end


    return counts

end


function chrf(
    reference::AbstractString,
    hypothesis::AbstractString;
    max_n::Int=6,
    beta::Float64=2.0
)

    reference =
        normalize_text(
            reference
        )


    hypothesis =
        normalize_text(
            hypothesis
        )


    precisions =
        Float64[]


    recalls =
        Float64[]


    for n in 1:max_n

        ref_ng =
            char_ngrams(
                reference,
                n
            )


        hyp_ng =
            char_ngrams(
                hypothesis,
                n
            )


        ref_total =
            sum(
                values(ref_ng);
                init=0
            )


        hyp_total =
            sum(
                values(hyp_ng);
                init=0
            )


        if ref_total == 0 ||
           hyp_total == 0

            continue

        end


        matched = sum(
            min(count, get(ref_ng, gram, 0))
            for (gram, count) in hyp_ng;
            init=0
        )


        push!(
            precisions,
            matched / hyp_total
        )


        push!(
            recalls,
            matched / ref_total
        )

    end


    isempty(precisions) &&
        return 0.0


    precision =
        mean(
            precisions
        )


    recall =
        mean(
            recalls
        )


    if precision + recall == 0

        return 0.0

    end


    return 100 *
           (1 + beta^2) *
           precision *
           recall /
           (
        beta^2 * precision +
        recall
    )

end


# ============================================================
# chrF DE CORPUS
# ============================================================

function corpus_chrf(
    references::Vector{String},
    hypotheses::Vector{String}
)

    reference =
        join(
            references,
            "\n"
        )


    hypothesis =
        join(
            hypotheses,
            "\n"
        )


    return chrf(
        reference,
        hypothesis
    )

end


# ============================================================
# CARGAR asr_per_audio.csv
# ============================================================

function load_asr_csv(
    path::String
)

    lines =
        readlines(path)


    isempty(lines) &&
        error(
            "asr_per_audio.csv está vacío."
        )


    header =
        parse_csv_line(
            lines[1]
        )


    column =
        Dict(
            name => i
            for (i, name)
            in
            enumerate(header)
        )


    required =
        [
            "model",
            "repetition",
            "audio",
            "reference_en",
            "hypothesis_en"
        ]


    for name in required

        haskey(
            column,
            name
        ) ||
            error(
                "Falta la columna '$name' en asr_per_audio.csv"
            )

    end


    rows =
        ASRRow[]


    for line in lines[2:end]

        isempty(strip(line)) &&
            continue


        fields =
            parse_csv_line(
                line
            )


        if length(fields) < length(header)

            error(
                "Fila CSV incompleta:\n$line"
            )

        end


        repetition =
            parse(
                Int,
                fields[
                    column["repetition"]
                ]
            )


        # Utilizamos una única repetición ASR.
        repetition != ASR_REPETITION &&
            continue


        model =
            fields[
                column["model"]
            ]


        model ∉ ASR_MODELS &&
            continue


        audio =
            fields[
                column["audio"]
            ]


        reference_en =
            fields[
                column["reference_en"]
            ]


        hypothesis_en =
            fields[
                column["hypothesis_en"]
            ]


        push!(
            rows,
            ASRRow(
                model,
                repetition,
                audio,
                reference_en,
                hypothesis_en
            )
        )

    end


    return rows

end


# ============================================================
# OLLAMA
# ============================================================

function translate(
    text::AbstractString
)

    # --------------------------------------------------------
    # Prompt fijo para absolutamente todas las traducciones.
    #
    # Queremos evitar que el modelo resuma o modifique
    # información clínicamente relevante.
    # --------------------------------------------------------

    prompt = """
     Translate the following clinical text from English to Spanish.

     Preserve all medically relevant information exactly, including:
     - symptoms
     - anatomical locations
     - medications
     - allergies
     - negations
     - numbers
     - time expressions
     - severity

     Do not summarize.
     Do not explain.
     Do not add information.
     Output ONLY the Spanish translation.

     Text:
     $text
     """


    payload = """
      {
        "model":"$(json_escape(LLM_MODEL))",
        "prompt":"$(json_escape(prompt))",
        "stream":false,
        "options":{
          "temperature":$(TEMPERATURE),
          "seed":$(SEED)
        }
      }
      """


    cmd =
        Cmd(
            [
            "curl",
            "-s",
            "-S",
            "-X",
            "POST",
            OLLAMA_URL,
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-"
        ]
        )


    output =
        IOBuffer()


    run(
        pipeline(
            cmd;
            stdin=
            IOBuffer(
                payload
            ),
            stdout=
            output
        )
    )


    response =
        String(
            take!(
                output
            )
        )


    translated_text =
        strip(
            json_extract_string(
                response,
                "response"
            )
        )


    # --------------------------------------------------------
    # Ollama entrega estas duraciones en nanosegundos.
    # --------------------------------------------------------

    total_duration_ns =
        Float64(
            json_extract_number(
                response,
                "total_duration";
                default=0
            )
        )


    load_duration_ns =
        Float64(
            json_extract_number(
                response,
                "load_duration";
                default=0
            )
        )


    prompt_eval_duration_ns =
        Float64(
            json_extract_number(
                response,
                "prompt_eval_duration";
                default=0
            )
        )


    eval_duration_ns =
        Float64(
            json_extract_number(
                response,
                "eval_duration";
                default=0
            )
        )


    prompt_eval_count =
        Int(
            json_extract_number(
                response,
                "prompt_eval_count";
                default=0
            )
        )


    eval_count =
        Int(
            json_extract_number(
                response,
                "eval_count";
                default=0
            )
        )


    return (
        text=
        translated_text, total_duration_s=
        total_duration_ns / 1e9, load_duration_s=
        load_duration_ns / 1e9, prompt_eval_duration_s=
        prompt_eval_duration_ns / 1e9, eval_duration_s=
        eval_duration_ns / 1e9, prompt_eval_count=
        prompt_eval_count, eval_count=
        eval_count
    )

end


# ============================================================
# MAIN
# ============================================================

function main()

    println()

    println(
        "======================================================"
    )

    println(
        " R2 — BENCHMARK DE TRADUCCIÓN"
    )

    println(
        "======================================================"
    )

    println()


    println(
        "LLM:               ",
        LLM_MODEL
    )


    println(
        "Cuantización LLM:  ",
        LLM_QUANTIZATION
    )


    println(
        "Temperature:       ",
        TEMPERATURE
    )


    println(
        "Seed:              ",
        SEED
    )


    println(
        "Referencia ES:     ",
        GROUNDTRUTH_ES
    )


    println()


    # ========================================================
    # VALIDAR ARCHIVOS
    # ========================================================

    isfile(ASR_CSV) ||
        error(
            "No existe:\n$ASR_CSV"
        )


    isfile(GROUNDTRUTH_ES) ||
        error(
            "No existe:\n$GROUNDTRUTH_ES"
        )


    # ========================================================
    # CARGAR DATOS
    # ========================================================

    asr_rows =
        load_asr_csv(
            ASR_CSV
        )


    spanish_refs =
        read_groundtruth(
            GROUNDTRUTH_ES
        )


    # ========================================================
    # VALIDACIONES DEL CORPUS
    # ========================================================

    expected_cases =
        length(ASR_MODELS) *
        length(spanish_refs)


    println(
        "Audios groundtruth ES: ",
        length(spanish_refs)
    )


    println(
        "Casos ASR encontrados: ",
        length(asr_rows)
    )


    println(
        "Casos esperados:        ",
        expected_cases
    )


    println()


    length(spanish_refs) == 10 ||
        @warn(
            "Se esperaban exactamente 10 audios.",
            encontrados =
                length(spanish_refs)
        )


    length(asr_rows) == expected_cases ||
        error(
            "El número de filas ASR no coincide con " *
            "modelos × audios."
        )


    # ========================================================
    # EJECUTAR LAS 50 TRADUCCIONES
    # ========================================================

    results =
        TranslationRow[]


    total_cases =
        length(asr_rows)


    for (case_index, row) in enumerate(asr_rows)

        haskey(
            spanish_refs,
            row.audio
        ) ||
            error(
                "No existe referencia española para:\n$(row.audio)"
            )


        reference_es =
            spanish_refs[
                row.audio
            ]


        println(
            "------------------------------------------------------"
        )


        println(
            "Caso ",
            case_index,
            "/",
            total_cases
        )


        println(
            "ASR:   ",
            row.model
        )


        println(
            "Audio: ",
            row.audio
        )


        # ----------------------------------------------------
        # Traducimos SOLAMENTE la hipótesis ASR.
        # ----------------------------------------------------

        translation =
            translate(
                row.hypothesis_en
            )


        # ----------------------------------------------------
        # Se compara directamente con groundtruth_es.txt
        # ----------------------------------------------------

        score =
            chrf(
                reference_es,
                translation.text
            )


        println(
            "chrF:  ",
            round(
                score;
                digits=3
            )
        )


        println(
            "Tiempo LLM: ",
            round(
                translation.total_duration_s;
                digits=3
            ),
            " s"
        )


        println()


        push!(
            results,
            TranslationRow(
                row.model,
                row.audio,
                score,
                row.reference_en,
                row.hypothesis_en,
                reference_es,
                translation.text,
                translation.total_duration_s,
                translation.load_duration_s,
                translation.prompt_eval_duration_s,
                translation.eval_duration_s,
                translation.prompt_eval_count,
                translation.eval_count
            )
        )

    end


    # ========================================================
    # GUARDAR RESULTADOS POR AUDIO
    # ========================================================

    open(
        OUT_PER_AUDIO,
        "w"
    ) do io


        write_csv_row(
            io,
            [
                "asr_model",
                "audio",
                "chrf",
                "reference_en",
                "hypothesis_en",
                "reference_es",
                "hypothesis_es",
                "total_duration_s",
                "load_duration_s",
                "prompt_eval_duration_s",
                "eval_duration_s",
                "prompt_eval_count",
                "eval_count"
            ]
        )


        for row in results

            write_csv_row(
                io,
                [
                    row.asr_model,
                    row.audio,
                    row.chrf,
                    row.reference_en,
                    row.hypothesis_en,
                    row.reference_es,
                    row.hypothesis_es,
                    row.total_duration_s,
                    row.load_duration_s,
                    row.prompt_eval_duration_s,
                    row.eval_duration_s,
                    row.prompt_eval_count,
                    row.eval_count
                ]
            )

        end

    end


    # ========================================================
    # RESUMEN POR FORMATO ASR
    # ========================================================

    open(
        OUT_SUMMARY,
        "w"
    ) do io


        write_csv_row(
            io,
            [
                "asr_model",
                "audio_count",
                "chrf_mean_per_audio",
                "chrf_sd_per_audio",
                "chrf_corpus",
                "llm_total_duration_mean_s",
                "llm_total_duration_total_s"
            ]
        )


        for model in ASR_MODELS


            group =
                filter(
                    row ->
                        row.asr_model == model,
                    results
                )


            sort!(
                group;
                by=
                row -> row.audio
            )


            isempty(group) &&
                error(
                    "No hay resultados para $model"
                )


            scores =
                [
                    row.chrf
                    for row in group
                ]


            references =
                [
                    row.reference_es
                    for row in group
                ]


            hypotheses =
                [
                    row.hypothesis_es
                    for row in group
                ]


            durations =
                [
                    row.total_duration_s
                    for row in group
                ]


            mean_score =
                mean(
                    scores
                )


            sd_score =
                length(scores) > 1 ?
                std(scores) :
                0.0


            corpus_score =
                corpus_chrf(
                    references,
                    hypotheses
                )


            mean_duration =
                mean(
                    durations
                )


            total_duration =
                sum(
                    durations
                )


            write_csv_row(
                io,
                [
                    model,
                    length(group),
                    mean_score,
                    sd_score,
                    corpus_score,
                    mean_duration,
                    total_duration
                ]
            )

        end

    end


    # ========================================================
    # METADATA
    # ========================================================

    open(
        OUT_METADATA,
        "w"
    ) do io


        write_csv_row(
            io,
            [
                "field",
                "value"
            ]
        )


        write_csv_row(
            io,
            [
                "timestamp",
                Dates.format(
                    now(),
                    dateformat"yyyy-mm-ddTHH:MM:SS"
                )
            ]
        )


        write_csv_row(
            io,
            [
                "experiment",
                "R2_translation"
            ]
        )


        write_csv_row(
            io,
            [
                "llm_model",
                LLM_MODEL
            ]
        )


        write_csv_row(
            io,
            [
                "llm_quantization",
                LLM_QUANTIZATION
            ]
        )


        write_csv_row(
            io,
            [
                "temperature",
                TEMPERATURE
            ]
        )


        write_csv_row(
            io,
            [
                "seed",
                SEED
            ]
        )


        write_csv_row(
            io,
            [
                "target_language",
                TARGET_LANGUAGE
            ]
        )


        write_csv_row(
            io,
            [
                "asr_repetition",
                ASR_REPETITION
            ]
        )


        write_csv_row(
            io,
            [
                "audio_count",
                length(spanish_refs)
            ]
        )


        write_csv_row(
            io,
            [
                "asr_models",
                join(
                    ASR_MODELS,
                    ","
                )
            ]
        )


        write_csv_row(
            io,
            [
                "reference_file",
                basename(
                    GROUNDTRUTH_ES
                )
            ]
        )


        write_csv_row(
            io,
            [
                "metric",
                "chrF"
            ]
        )


        write_csv_row(
            io,
            [
                "chrf_max_n",
                6
            ]
        )


        write_csv_row(
            io,
            [
                "chrf_beta",
                2
            ]
        )


        write_csv_row(
            io,
            [
                "julia_version",
                VERSION
            ]
        )


        write_csv_row(
            io,
            [
                "os",
                Sys.KERNEL
            ]
        )

    end


    # ========================================================
    # MOSTRAR RESUMEN EN TERMINAL
    # ========================================================

    println()

    println(
        "======================================================"
    )

    println(
        " RESULTADOS FINALES R2"
    )

    println(
        "======================================================"
    )

    println()


    for model in ASR_MODELS


        group =
            filter(
                row ->
                    row.asr_model == model,
                results
            )


        sort!(
            group;
            by=
            row -> row.audio
        )


        scores =
            [
                row.chrf
                for row in group
            ]


        refs =
            [
                row.reference_es
                for row in group
            ]


        hyps =
            [
                row.hypothesis_es
                for row in group
            ]


        println(
            rpad(
                model,
                8
            ),
            "chrF medio = ",
            lpad(
                round(
                    mean(scores);
                    digits=3
                ),
                7
            ),
            "   chrF corpus = ",
            lpad(
                round(
                    corpus_chrf(
                        refs,
                        hyps
                    );
                    digits=3
                ),
                7
            )
        )

    end


    println()

    println(
        "Archivos creados:"
    )


    println(
        "  ",
        OUT_PER_AUDIO
    )


    println(
        "  ",
        OUT_SUMMARY
    )


    println(
        "  ",
        OUT_METADATA
    )


    println()

    println(
        "Traducciones realizadas: ",
        length(results)
    )


    println()

    println(
        "======================================================"
    )

end


main()