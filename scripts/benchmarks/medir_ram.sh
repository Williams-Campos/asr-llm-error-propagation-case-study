#!/bin/bash
# Medición de RAM (RSS) por formato de cuantización de Whisper.
# Ejecutar desde la raíz del proyecto: bash scripts/benchmarks/medir_ram.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

AUDIO="$PROJECT_ROOT/data/audio/audio01(dolor toracico).wav"
WHISPER="$PROJECT_ROOT/whisper.cpp/build/bin/whisper-cli"

MODELOS=(
    "f32:$PROJECT_ROOT/whisper.cpp/models/ggml-base-f32.bin"
    "f16:$PROJECT_ROOT/whisper.cpp/models/ggml-base-f16.bin"
    "q8_0:$PROJECT_ROOT/whisper.cpp/models/ggml-base-q8_0.bin"
    "q5_1:$PROJECT_ROOT/whisper.cpp/models/ggml-base-q5_1.bin"
    "q4_0:$PROJECT_ROOT/whisper.cpp/models/ggml-base-q4_0.bin"
)

REPETICIONES=5

echo "modelo,repeticion,rss_bytes,rss_mib" > "$PROJECT_ROOT/results/csv/ram_resultados.csv"

for item in "${MODELOS[@]}"; do

    nombre="${item%%:*}"
    modelo="${item#*:}"

    echo
    echo "======================================"
    echo "Modelo: $nombre"
    echo "======================================"

    for ((i=1; i<=REPETICIONES; i++)); do

        archivo_tmp=$(mktemp)

        /usr/bin/time -l "$WHISPER" \
            -m "$modelo" \
            -f "$AUDIO" \
            -t 4 \
            > /dev/null 2> "$archivo_tmp"

        rss=$(grep "maximum resident set size" "$archivo_tmp" | awk '{print $1}')

        if [ -z "$rss" ]; then
            echo "ERROR: no se pudo obtener RSS para $nombre repetición $i"
            cat "$archivo_tmp"
            rm "$archivo_tmp"
            exit 1
        fi

        rss_mib=$(awk -v r="$rss" 'BEGIN {printf "%.2f", r/1048576}')

        echo "Run $i: $rss bytes = $rss_mib MiB"

        echo "$nombre,$i,$rss,$rss_mib" >> "$PROJECT_ROOT/results/csv/ram_resultados.csv"

        rm "$archivo_tmp"
    done

done

echo
echo "Resultados guardados en results/csv/ram_resultados.csv"
