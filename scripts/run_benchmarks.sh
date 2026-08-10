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
#
#  Le dimensioni sono sovrascrivibili da ambiente, utile per una campagna
#  ridotta su macchine senza GPU (dove la baseline seriale domina i tempi):
#
#    REPS=1 ITER=20 N_LIST="10000 50000 100000" ./scripts/run_benchmarks.sh
# ===========================================================================

set -euo pipefail

BIN="${1:-build/kmeans_bench}"
OUT_DIR="results"
CSV="${OUT_DIR}/benchmarks.csv"
REPS="${REPS:-3}"
ITER="${ITER:-50}"

# Dimensioni degli sweep e configurazione di riferimento (n, d, k fissi negli
# sweep in cui non variano). Tutte sovrascrivibili da ambiente.
N_LIST="${N_LIST:-10000 50000 100000 250000 500000 1000000 2000000}"
K_LIST="${K_LIST:-4 8 16 32 64 128 256}"
D_LIST="${D_LIST:-2 4 8 16 32 64 128}"
REF_N="${REF_N:-500000}"
REF_D="${REF_D:-32}"
REF_K="${REF_K:-32}"

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
echo "=== 1/4  Scalabilita' in N (d=${REF_D}, k=${REF_K}) ==="
for N in ${N_LIST}; do
    echo "--- n=${N}"
    "${BIN}" --n "${N}" --d "${REF_D}" --k "${REF_K}" --iter "${ITER}" --reps "${REPS}" \
             --impl "${IMPLS}" --csv "${CSV}"
done

echo
echo "=== 2/4  Scalabilita' in K (n=${REF_N}, d=${REF_D}) ==="
for K in ${K_LIST}; do
    echo "--- k=${K}"
    "${BIN}" --n "${REF_N}" --d "${REF_D}" --k "${K}" --iter "${ITER}" --reps "${REPS}" \
             --impl "${IMPLS}" --csv "${CSV}"
done

echo
echo "=== 3/4  Scalabilita' in D (n=${REF_N}, k=${REF_K}) ==="
for D in ${D_LIST}; do
    echo "--- d=${D}"
    "${BIN}" --n "${REF_N}" --d "${D}" --k "${REF_K}" --iter "${ITER}" --reps "${REPS}" \
             --impl "${IMPLS}" --csv "${CSV}"
done

echo
echo "=== 4/4  Scalabilita' forte OpenMP (n=${REF_N}, d=${REF_D}, k=${REF_K}) ==="
MAX_THREADS="$(nproc)"
THREAD_LIST=""
T=1
while [[ "${T}" -lt "${MAX_THREADS}" ]]; do
    THREAD_LIST="${THREAD_LIST} ${T}"
    T=$((T * 2))
done
THREAD_LIST="${THREAD_LIST} ${MAX_THREADS}"

for T in ${THREAD_LIST}; do
    echo "--- threads=${T}"
    "${BIN}" --n "${REF_N}" --d "${REF_D}" --k "${REF_K}" --iter "${ITER}" --reps "${REPS}" \
             --impl omp --threads "${T}" --csv "${OUT_DIR}/omp_scaling.csv"
done

echo
echo "Fatto. Risultati in ${CSV} e ${OUT_DIR}/omp_scaling.csv"
echo "Genera i grafici con: python3 scripts/plot_results.py"
