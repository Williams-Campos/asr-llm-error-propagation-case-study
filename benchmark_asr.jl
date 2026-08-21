#!/usr/bin/env julia
# Experimento final ASR: 5 formatos × 10 WAV en inglés × N repeticiones.
# Uso: julia benchmark_asr.jl [repeticiones=5]

using Dates
using Statistics
using Printf

const ROOT = @__DIR__
const CLI = joinpath(ROOT, "whisper.cpp", "build", "bin", "whisper-cli")
const MODELS_DIR = joinpath(ROOT, "whisper.cpp", "models")
const AUDIO_DIR = joinpath(ROOT, "audio")
const GROUNDTRUTH = joinpath(ROOT, "groundtruth.txt")
const OUT_DIR = joinpath(ROOT, "out", "asr")
const RESULTS_DIR = joinpath(ROOT, "results")
const LANGUAGE = "en"
const THREADS, BEAM_SIZE, BEST_OF = 4, 5, 5
const TEMPERATURE, TEMPERATURE_INCREMENT = 0.0, 0.0

# Modifica estas rutas solo si tus modelos fueron guardados con otro nombre.
const MODELS = [
    ("F32",  joinpath(MODELS_DIR, "ggml-base-f32.bin")),
    ("F16",  joinpath(MODELS_DIR, "ggml-base-f16.bin")),
    ("Q8_0", joinpath(MODELS_DIR, "ggml-base-q8_0.bin")),
    ("Q5_1", joinpath(MODELS_DIR, "ggml-base-q5_1.bin")),
    ("Q4_0", joinpath(MODELS_DIR, "ggml-base-q4_0.bin")),
]

function usage_and_exit()
    println(stderr, "uso: julia benchmark_asr.jl [repeticiones=5]")
    exit(1)
end

csv_escape(x) = "\"" * replace(string(x), "\"" => "\"\"") * "\""
csv_num(x) = @sprintf("%.9f", x)

function write_csv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ','))
        foreach(row -> println(io, join(row, ',')), rows)
    end
end

function read_groundtruth(path)
    refs = Dict{String,String}()
    for line in eachline(path)
        isempty(strip(line)) && continue
        m = match(r"^([^,]+),\"(.*)\"$", line)
        m === nothing && error("Línea inválida en $path: $line")
        refs[String(m.captures[1])] = String(m.captures[2])
    end
    refs
end

function wav_duration_seconds(path)
    open(path, "r") do io
        String(read(io, 4)) == "RIFF" || error("No es WAV RIFF: $path")
        read(io, 8) # tamaño RIFF y firma WAVE
        byte_rate = 0; data_size = 0
        while !eof(io)
            id = String(read(io, 4)); length(id) == 4 || break
            n = Int(ltoh(read(io, UInt32)))
            if id == "fmt "
                fmt = read(io, n)
                length(fmt) >= 12 || error("Chunk fmt inválido: $path")
                byte_rate = Int(ltoh(reinterpret(UInt32, fmt[9:12])[1]))
            elseif id == "data"
                data_size = n; skip(io, n)
            else
                skip(io, n)
            end
            isodd(n) && !eof(io) && skip(io, 1)
        end
        byte_rate > 0 && data_size > 0 || error("No se pudo obtener duración de $path")
        data_size / byte_rate
    end
end

function normalize(text)
    t = lowercase(text)
    t = replace(t, r"[^\p{L}\p{N}\s]" => "")
    strip(replace(t, r"\s+" => " "))
end

function levenshtein(a::AbstractVector, b::AbstractVector)
    prev = collect(0:length(b)); curr = zeros(Int, length(b) + 1)
    for i in eachindex(a)
        curr[1] = i
        for j in eachindex(b)
            cost = a[i] == b[j] ? 0 : 1
            curr[j + 1] = min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost)
        end
        prev, curr = curr, prev
    end
    prev[end]
end

function error_counts(reference, hypothesis, unit::Symbol)
    ref = unit == :word ? split(normalize(reference)) : collect(normalize(reference))
    hyp = unit == :word ? split(normalize(hypothesis)) : collect(normalize(hypothesis))
    isempty(ref) && error("Referencia vacía")
    levenshtein(ref, hyp), length(ref)
end

function run_transcription(model, wav, prefix)
    mkpath(dirname(prefix))
    # -nf y -tpi 0 bloquean el fallback de temperatura; el resto queda fijo.
    cmd = `$CLI -m $model -f $wav -l $LANGUAGE -t $THREADS -bs $BEAM_SIZE -bo $BEST_OF -tp $TEMPERATURE -tpi $TEMPERATURE_INCREMENT -nf -nt -np -otxt -of $prefix`
    elapsed = @elapsed run(pipeline(cmd; stdout=devnull, stderr=devnull))
    text_file = prefix * ".txt"
    isfile(text_file) || error("whisper-cli no creó $text_file")
    strip(read(text_file, String)), elapsed
end

function git_commit()
    try
        strip(read(`git -C $(joinpath(ROOT, "whisper.cpp")) rev-parse HEAD`, String))
    catch
        "no_disponible"
    end
end

function main()
    length(ARGS) <= 1 || usage_and_exit()
    repetitions = isempty(ARGS) ? 5 : try parse(Int, ARGS[1]) catch; usage_and_exit() end
    repetitions > 0 || error("repeticiones debe ser positiva")
    isfile(CLI) || error("No existe whisper-cli: $CLI")
    missing = ["$label ($path)" for (label, path) in MODELS if !isfile(path)]
    isempty(missing) || error("Faltan modelos:\n  " * join(missing, "\n  "))

    refs = read_groundtruth(GROUNDTRUTH)
    wavs = sort(filter(f -> endswith(lowercase(f), ".wav"), readdir(AUDIO_DIR)))
    length(wavs) == 10 || error("Se esperaban 10 WAV; se encontraron $(length(wavs))")
    all(haskey(refs, w) for w in wavs) || error("Faltan referencias en $GROUNDTRUTH")
    mkpath(OUT_DIR); mkpath(RESULTS_DIR)

    metadata = [
        ["timestamp", csv_escape(Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"))], ["language", LANGUAGE],
        ["threads", string(THREADS)], ["beam_size", string(BEAM_SIZE)], ["best_of", string(BEST_OF)],
        ["temperature", string(TEMPERATURE)], ["temperature_increment", string(TEMPERATURE_INCREMENT)],
        ["temperature_fallback", "disabled (-nf)"], ["gpu", "CLI default: enabled"],
        ["whisper_commit", git_commit()], ["julia_version", string(VERSION)], ["os", string(Sys.KERNEL)],
        ["cpu_threads", string(Sys.CPU_THREADS)], ["repetitions", string(repetitions)], ["audio_count", "10"],
    ]
    write_csv(joinpath(RESULTS_DIR, "asr_metadata.csv"), ["field", "value"], metadata)

    per_audio = Vector{Vector{String}}(); summary = Vector{Vector{String}}()
    println("ASR final: 5 modelos × 10 audios × $repetitions repeticiones")
    for (label, model) in MODELS
        for repetition in 1:repetitions
            werr = wref = cerr = cref = 0; rtfs = Float64[]; times = Float64[]; durations = Float64[]
            for wav_name in wavs
                wav = joinpath(AUDIO_DIR, wav_name); duration = wav_duration_seconds(wav)
                base = splitext(wav_name)[1]
                prefix = joinpath(OUT_DIR, label, "run_" * lpad(repetition, 2, '0'), base)
                hypothesis, elapsed = run_transcription(model, wav, prefix)
                we, wn = error_counts(refs[wav_name], hypothesis, :word)
                ce, cn = error_counts(refs[wav_name], hypothesis, :char)
                rtf = elapsed / duration
                push!(per_audio, [csv_escape(label), csv_escape(abspath(model)), string(repetition), csv_escape(wav_name), csv_num(duration), csv_num(elapsed), csv_num(rtf), string(we), string(wn), csv_num(we / wn), string(ce), string(cn), csv_num(ce / cn), csv_escape(refs[wav_name]), csv_escape(hypothesis)])
                werr += we; wref += wn; cerr += ce; cref += cn
                push!(rtfs, rtf); push!(times, elapsed); push!(durations, duration)
            end
            push!(summary, [csv_escape(label), csv_escape(abspath(model)), string(repetition), "10", string(werr), string(wref), csv_num(werr / wref), string(cerr), string(cref), csv_num(cerr / cref), csv_num(sum(times)), csv_num(sum(durations)), csv_num(sum(times) / sum(durations)), csv_num(mean(rtfs)), csv_num(std(rtfs))])
            println("  $label, repetición $repetition/$repetitions: WER corpus=$(round(werr / wref; digits=4)), RTF corpus=$(round(sum(times) / sum(durations); digits=4))")
        end
    end
    write_csv(joinpath(RESULTS_DIR, "asr_per_audio.csv"), ["model", "model_path", "repetition", "audio", "audio_seconds", "inference_seconds", "rtf", "word_errors", "reference_words", "wer", "char_errors", "reference_characters", "cer", "reference_en", "hypothesis_en"], per_audio)
    write_csv(joinpath(RESULTS_DIR, "asr_summary.csv"), ["model", "model_path", "repetition", "audio_count", "word_errors", "reference_words", "wer_corpus", "char_errors", "reference_characters", "cer_corpus", "inference_seconds_total", "audio_seconds_total", "rtf_corpus", "rtf_mean_per_audio", "rtf_sd_per_audio"], summary)
    println("\nListo. CSV: $(joinpath(RESULTS_DIR, "asr_per_audio.csv")) y asr_summary.csv")
end

main()
