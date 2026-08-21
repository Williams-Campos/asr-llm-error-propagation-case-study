#!/usr/bin/env julia

using Statistics
using Dates


# ============================================================
# R3 — PROPAGACIÓN DEL ERROR EN LA CASCADA ASR -> LLM
#
# Se evalúan dos cuantizaciones del MISMO LLM:
#
#   gemma3:1b-it-q8_0  -> Q8_0
#   gemma3:1b          -> Q4_K_M
#
# Para cada LLM se prueban:
#
#   1. REFERENCE -> LLM
#   2. F32       -> LLM
#   3. Q8_0      -> LLM
#   4. Q5_1      -> LLM
#   5. Q4_0      -> LLM
#
# Todas las traducciones se comparan directamente contra:
#
#   groundtruth_es.txt
#
# Total:
#
#   2 LLM × 5 condiciones × 10 audios = 100 traducciones
# ============================================================


# ============================================================
# CONFIGURACIÓN
# ============================================================

const BASE_DIR = @__DIR__

const ASR_CSV =
    joinpath(BASE_DIR, "asr_per_audio.csv")

const GROUNDTRUTH_ES =
    joinpath(BASE_DIR, "groundtruth_es.txt")

const OUT_PER_AUDIO =
    joinpath(BASE_DIR, "r3_per_audio.csv")

const OUT_SUMMARY =
    joinpath(BASE_DIR, "r3_summary.csv")

const OUT_METADATA =
    joinpath(BASE_DIR, "r3_metadata.csv")


# ============================================================
# OLLAMA
# ============================================================

const OLLAMA_URL = "http://localhost:11434/api/generate"


# ============================================================
# DOS CUANTIZACIONES DEL MISMO GEMMA 3 1B
# ============================================================

const LLM_MODELS = [
    (
        name="gemma3:1b-it-q8_0",
        quantization="Q8_0"
    ),
    (
        name="gemma3:1b",
        quantization="Q4_K_M"
    )
]


# ============================================================
# MODELOS ASR
# ============================================================

const ASR_MODELS = [
    "F32",
    "Q8_0",
    "Q5_1",
    "Q4_0"
]


# ============================================================
# PARÁMETROS EXPERIMENTALES
# ============================================================

const TEMPERATURE = 0.0

const SEED = 42

const TARGET_LANGUAGE = "Spanish"

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


struct ExperimentRow
    llm_model::String
    llm_quantization::String

    condition::String
    asr_model::String

    audio::String

    chrf::Float64

    input_en::String
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
# CSV
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
# Formato:
#
# audio01.wav,"Texto español..."
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


        m === nothing &&
            error(
                "Línea inválida en groundtruth_es.txt:\n$line"
            )


        filename =
            String(
                m.captures[1]
            )


        text =
            String(
                m.captures[2]
            )


        haskey(refs, filename) &&
            error(
                "Audio repetido en groundtruth_es.txt: $filename"
            )


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


    return String(take!(io))
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
            "Campo '$key' no encontrado en respuesta de Ollama:\n$json"
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

                indices = Int[]

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

            print(io, c)

        end


        i =
            nextind(
                json,
                i
            )
    end


    return String(take!(io))
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
                chars[i:i+n-1]
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
        normalize_text(reference)


    hypothesis =
        normalize_text(hypothesis)


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
            min(
                count,
                get(
                    ref_ng,
                    gram,
                    0
                )
            )
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


    p =
        mean(precisions)


    r =
        mean(recalls)


    if p + r == 0
        return 0.0
    end


    return 100 *
           (1 + beta^2) *
           p *
           r /
           (
        beta^2 * p +
        r
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
# CARGAR ASR
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
            for (i, name) in enumerate(header)
        )


    required = [
        "model",
        "repetition",
        "audio",
        "reference_en",
        "hypothesis_en"
    ]


    for name in required

        haskey(column, name) ||
            error(
                "Falta columna '$name' en asr_per_audio.csv"
            )
    end


    rows =
        ASRRow[]


    for line in lines[2:end]

        isempty(strip(line)) &&
            continue


        fields =
            parse_csv_line(line)


        length(fields) < length(header) &&
            error(
                "Fila incompleta en asr_per_audio.csv:\n$line"
            )


        repetition =
            parse(
                Int,
                fields[
                    column["repetition"]
                ]
            )


        repetition != ASR_REPETITION &&
            continue


        model =
            fields[
                column["model"]
            ]


        model ∉ ASR_MODELS &&
            continue


        push!(
            rows,
            ASRRow(
                model,
                repetition,
                fields[
                    column["audio"]
                ],
                fields[
                    column["reference_en"]
                ],
                fields[
                    column["hypothesis_en"]
                ]
            )
        )
    end


    return rows
end


# ============================================================
# REFERENCIA INGLESA POR AUDIO
# ============================================================

function build_english_refs(
    rows::Vector{ASRRow}
)

    refs =
        Dict{String,String}()


    for row in rows

        if haskey(refs, row.audio)

            refs[row.audio] == row.reference_en ||
                error(
                    "Referencia inglesa inconsistente para $(row.audio)"
                )

        else

            refs[row.audio] =
                row.reference_en
        end
    end


    return refs
end


# ============================================================
# HIPÓTESIS ASR POR MODELO Y AUDIO
# ============================================================

function build_asr_hypotheses(
    rows::Vector{ASRRow}
)

    hypotheses =
        Dict{
            Tuple{String,String},
            String
        }()


    for row in rows

        key =
            (
                row.model,
                row.audio
            )


        haskey(hypotheses, key) &&
            error(
                "Hipótesis duplicada para $key"
            )


        hypotheses[key] =
            row.hypothesis_en
    end


    return hypotheses
end


# ============================================================
# OLLAMA
# ============================================================

function translate(
    text::AbstractString,
    model::String
)

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
        "model":"$(json_escape(model))",
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
            IOBuffer(payload),
            stdout=
            output
        )
    )


    response =
        String(
            take!(output)
        )


    translated_text =
        strip(
            json_extract_string(
                response,
                "response"
            )
        )


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
# WARM-UP
#
# Se hace una llamada inicial que NO entra en las métricas.
# Así evitamos que el tiempo de carga inicial contamine
# la comparación de latencia entre los dos LLM.
# ============================================================

function warmup(model::String)

    println(
        "Warm-up de ",
        model,
        "..."
    )


    translate(
        "The patient is stable.",
        model
    )


    println(
        "Warm-up completado."
    )
end


# ============================================================
# EJECUTAR UN CASO
# ============================================================

function run_case(
    llm,
    condition::String,
    asr_model::String,
    audio::String,
    input_en::String,
    reference_es::String
)

    translation =
        translate(
            input_en,
            llm.name
        )


    score =
        chrf(
            reference_es,
            translation.text
        )


    println(
        rpad(llm.quantization, 9),
        " | ",
        rpad(condition, 20),
        " | ",
        rpad(asr_model, 9),
        " | ",
        rpad(audio, 34),
        " | chrF = ",
        round(
            score;
            digits=3
        ),
        " | ",
        round(
            translation.total_duration_s;
            digits=3
        ),
        " s"
    )


    return ExperimentRow(
        llm.name,
        llm.quantization,
        condition,
        asr_model,
        audio,
        score,
        input_en,
        reference_es,
        translation.text,
        translation.total_duration_s,
        translation.load_duration_s,
        translation.prompt_eval_duration_s,
        translation.eval_duration_s,
        translation.prompt_eval_count,
        translation.eval_count
    )
end


# ============================================================
# MAIN
# ============================================================

function main()

    println()

    println(
        "============================================================"
    )

    println(
        " R3 — PROPAGACIÓN DEL ERROR ASR -> LLM"
    )

    println(
        "============================================================"
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


    english_refs =
        build_english_refs(
            asr_rows
        )


    asr_hypotheses =
        build_asr_hypotheses(
            asr_rows
        )


    audios =
        sort(
            collect(
                keys(english_refs)
            )
        )


    # ========================================================
    # VALIDACIONES
    # ========================================================

    length(audios) == 10 ||
        error(
            "Se esperaban 10 audios y se encontraron $(length(audios))."
        )


    length(spanish_refs) == 10 ||
        error(
            "groundtruth_es.txt debe contener exactamente 10 audios."
        )


    for audio in audios

        haskey(
            spanish_refs,
            audio
        ) ||
            error(
                "No existe referencia española para '$audio'."
            )
    end


    expected_asr_cases =
        length(ASR_MODELS) *
        length(audios)


    length(asr_rows) == expected_asr_cases ||
        error(
            "Se esperaban $expected_asr_cases filas ASR " *
            "para la repetición $ASR_REPETITION, " *
            "pero se encontraron $(length(asr_rows))."
        )


    println(
        "Audios:             ",
        length(audios)
    )


    println(
        "Modelos LLM:        ",
        length(LLM_MODELS)
    )


    println(
        "Entradas por LLM:   ",
        1 + length(ASR_MODELS)
    )


    println(
        "Casos totales:      ",
        length(LLM_MODELS) *
        (1 + length(ASR_MODELS)) *
        length(audios)
    )


    println(
        "Temperature:        ",
        TEMPERATURE
    )


    println(
        "Seed:               ",
        SEED
    )


    println(
        "Referencia ES:      ",
        basename(GROUNDTRUTH_ES)
    )


    println()


    results =
        ExperimentRow[]


    # ========================================================
    # EXPERIMENTO
    # ========================================================

    for llm in LLM_MODELS

        println()

        println(
            "============================================================"
        )

        println(
            " LLM: ",
            llm.name,
            " — ",
            llm.quantization
        )

        println(
            "============================================================"
        )


        # ----------------------------------------------------
        # Warm-up no contabilizado.
        # ----------------------------------------------------

        warmup(
            llm.name
        )


        println()


        # ====================================================
        # CONDICIÓN 1:
        #
        # Referencia EN -> LLM
        #
        # Error aislado del LLM
        # ====================================================

        for audio in audios

            row =
                run_case(
                    llm,
                    "llm_only",
                    "REFERENCE",
                    audio,
                    english_refs[audio],
                    spanish_refs[audio]
                )


            push!(
                results,
                row
            )
        end


        # ====================================================
        # CONDICIÓN 2:
        #
        # F32 -> LLM
        #
        # ASR alta precisión
        #
        # CONDICIÓN 3:
        #
        # Q8/Q5/Q4 -> LLM
        #
        # Error compuesto
        # ====================================================

        for asr_model in ASR_MODELS

            condition =
                if asr_model == "F32"
                    "asr_high_precision"
                else
                    "asr_quantized"
                end


            for audio in audios

                key =
                    (
                        asr_model,
                        audio
                    )


                haskey(
                    asr_hypotheses,
                    key
                ) ||
                    error(
                        "No existe hipótesis ASR para $key."
                    )


                row =
                    run_case(
                        llm,
                        condition,
                        asr_model,
                        audio,
                        asr_hypotheses[key],
                        spanish_refs[audio]
                    )


                push!(
                    results,
                    row
                )
            end
        end
    end


    # ========================================================
    # VALIDAR 100 CASOS
    # ========================================================

    expected_total =
        length(LLM_MODELS) *
        (1 + length(ASR_MODELS)) *
        length(audios)


    length(results) == expected_total ||
        error(
            "Se esperaban $expected_total resultados, " *
            "pero se generaron $(length(results))."
        )


    # ========================================================
    # r3_per_audio.csv
    # ========================================================

    open(
        OUT_PER_AUDIO,
        "w"
    ) do io

        write_csv_row(
            io,
            [
                "llm_model",
                "llm_quantization",
                "condition",
                "asr_model",
                "audio",
                "chrf",
                "input_en",
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
                    row.llm_model,
                    row.llm_quantization,
                    row.condition,
                    row.asr_model,
                    row.audio,
                    row.chrf,
                    row.input_en,
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
    # r3_summary.csv
    # ========================================================

    open(
        OUT_SUMMARY,
        "w"
    ) do io

        write_csv_row(
            io,
            [
                "llm_model",
                "llm_quantization",
                "condition",
                "asr_model",
                "audio_count",
                "chrf_mean_per_audio",
                "chrf_sd_per_audio",
                "chrf_corpus",
                "total_duration_mean_s",
                "total_duration_total_s",
                "eval_duration_mean_s"
            ]
        )


        for llm in LLM_MODELS

            input_models = [
                "REFERENCE",
                "F32",
                "Q8_0",
                "Q5_1",
                "Q4_0"
            ]


            for input_model in input_models

                group =
                    filter(
                        row ->
                            row.llm_model == llm.name &&
                                row.asr_model == input_model,
                        results
                    )


                isempty(group) &&
                    error(
                        "No existen resultados para " *
                        "$(llm.name) / $input_model"
                    )


                sort!(
                    group;
                    by=row -> row.audio
                )


                scores = [
                    row.chrf
                    for row in group
                ]


                refs = [
                    row.reference_es
                    for row in group
                ]


                hyps = [
                    row.hypothesis_es
                    for row in group
                ]


                total_durations = [
                    row.total_duration_s
                    for row in group
                ]


                eval_durations = [
                    row.eval_duration_s
                    for row in group
                ]


                condition =
                    if input_model == "REFERENCE"

                        "llm_only"

                    elseif input_model == "F32"

                        "asr_high_precision"

                    else

                        "asr_quantized"
                    end


                write_csv_row(
                    io,
                    [
                        llm.name,
                        llm.quantization,
                        condition,
                        input_model,
                        length(group),
                        mean(scores),
                        length(scores) > 1 ?
                        std(scores) :
                        0.0,
                        corpus_chrf(
                            refs,
                            hyps
                        ),
                        mean(
                            total_durations
                        ),
                        sum(
                            total_durations
                        ),
                        mean(
                            eval_durations
                        )
                    ]
                )
            end
        end
    end


    # ========================================================
    # r3_metadata.csv
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
                "R3_error_propagation"
            ]
        )


        write_csv_row(
            io,
            [
                "llm_1_model",
                LLM_MODELS[1].name
            ]
        )


        write_csv_row(
            io,
            [
                "llm_1_quantization",
                LLM_MODELS[1].quantization
            ]
        )


        write_csv_row(
            io,
            [
                "llm_2_model",
                LLM_MODELS[2].name
            ]
        )


        write_csv_row(
            io,
            [
                "llm_2_quantization",
                LLM_MODELS[2].quantization
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
                length(audios)
            ]
        )


        write_csv_row(
            io,
            [
                "conditions",
                "REFERENCE,F32,Q8_0,Q5_1,Q4_0"
            ]
        )


        write_csv_row(
            io,
            [
                "reference_es_file",
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
                "warmup",
                "one unmeasured request per LLM"
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
    # MOSTRAR RESUMEN FINAL
    # ========================================================

    println()

    println(
        "============================================================"
    )

    println(
        " R3 COMPLETADO"
    )

    println(
        "============================================================"
    )

    println()


    println(
        "Resultados generados:"
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
        "Casos ejecutados: ",
        length(results)
    )


    println()

    println(
        "RESUMEN chrF corpus:"
    )

    println()


    for llm in LLM_MODELS

        println(
            llm.name,
            " (",
            llm.quantization,
            ")"
        )


        for input_model in [
            "REFERENCE",
            "F32",
            "Q8_0",
            "Q5_1",
            "Q4_0"
        ]

            group =
                filter(
                    row ->
                        row.llm_model == llm.name &&
                            row.asr_model == input_model,
                    results
                )


            sort!(
                group;
                by=row -> row.audio
            )


            refs = [
                row.reference_es
                for row in group
            ]


            hyps = [
                row.hypothesis_es
                for row in group
            ]


            score =
                corpus_chrf(
                    refs,
                    hyps
                )


            println(
                "  ",
                rpad(
                    input_model,
                    10
                ),
                " -> chrF = ",
                round(
                    score;
                    digits=3
                )
            )
        end


        println()
    end


    println(
        "============================================================"
    )

end


main()