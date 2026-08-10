#!/usr/bin/env bash
# ===========================================================================
#  Campagna di benchmark completa.
#
#  Produce results/benchmarks.csv con tre studi di scalabilita' separati, che
#  sono esattamente i tre grafici della sezione "Profiling" del report:
#
#    1. scalabilita' in N  (k e d fissi)  -> lo speedup GPU deve crescere con N
#    2. scalabilita' in K  (n e d fissi)  -> aumenta il lavoro aritmetico per punto
#    3. scalabilita' in D  (n e k fissi)  -> aumenta il traffico di memoria per punto
#    4. scalabilita' forte OpenMP         -> speedup al variare del numero di thread
#
#  Uso:  ./scripts/run_benchmarks.sh [percorso_eseguibile]
# ===========================================================================

set -euo pipefail

BIN="${1:-build/kmeans_bench}"
OUT_DIR="results"
CSV="${OUT_DIR}/benchmarks.csv"
REPS="${REPS:-3}"
ITER="${ITER:-50}"

if [[ ! -x "${BIN}" ]]; then
    echo "Eseguibile non trovato: ${BIN}" >&2
    echo "Compila prima con 'make' oppure 'make cuda'." >&2
    exit 1
fi

# Se il binario include CUDA lo si capisce dal banner che stampa all'avvio.
if "${BIN}" --n 1000 --d 2 --k 2 --reps 0 --impl serial 2>/dev/null | grep -q "^GPU"; then
    IMPLS="serial,omp,cuda_naive,cuda_opt"
    echo "Build con CUDA rilevata: verranno misurate tutte e quattro le versioni."
else
    IMPLS="serial,omp"
    echo "Build senza CUDA: verranno misurate solo le versioni CPU."
fi

mkdir -p "${OUT_DIR}"
rm -f "${CSV}"

echo
echo "=== 1/4  Scalabilita' in N (d=32, k=32) ==="
for N in 10000 50000 100000 250000 500000 1000000 2000000; do
    echo "--- n=${N}"
    "${BIN}" --n "${N}" --d 32 --k 32 --iter "${ITER}" --reps "${REPS}" \
             --impl "${IMPLS}" --csv "${CSV}"
done

echo
echo "=== 2/4  Scalabilita' in K (n=500000, d=32) ==="
for K in 4 8 16 32 64 128 256; do
    echo "--- k=${K}"
    "${BIN}" --n 500000 --d 32 --k "${K}" --iter "${ITER}" --reps "${REPS}" \
             --impl "${IMPLS}" --csv "${CSV}"
done

echo
echo "=== 3/4  Scalabilita' in D (n=500000, k=32) ==="
for D in 2 4 8 16 32 64 128; do
    echo "--- d=${D}"
    "${BIN}" --n 500000 --d "${D}" --k 32 --iter "${ITER}" --reps "${REPS}" \
             --impl "${IMPLS}" --csv "${CSV}"
done

echo
echo "=== 4/4  Scalabilita' forte OpenMP (n=500000, d=32, k=32) ==="
MAX_THREADS="$(nproc)"
T=1
while [[ "${T}" -le "${MAX_THREADS}" ]]; do
    echo "--- threads=${T}"
    "${BIN}" --n 500000 --d 32 --k 32 --iter "${ITER}" --reps "${REPS}" \
             --impl omp --threads "${T}" --csv "${OUT_DIR}/omp_scaling.csv"
    T=$((T * 2))
done

echo
echo "Fatto. Risultati in ${CSV} e ${OUT_DIR}/omp_scaling.csv"
echo "Genera i grafici con: python3 scripts/plot_results.py"
