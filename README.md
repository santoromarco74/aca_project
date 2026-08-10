# K-means clustering: CPU seriale, OpenMP e CUDA

Progetto per il corso di **Advanced Computer Architecture**.

Implementazione dell'algoritmo di Lloyd per il clustering K-means in quattro
versioni, confrontate su correttezza, tempo di esecuzione e comportamento
rilevato dal profiler:

| Versione | File | Sintesi |
|---|---|---|
| Seriale | `src/kmeans_serial.cpp` | Baseline di riferimento, layout row-major, `-O3` |
| OpenMP | `src/kmeans_omp.cpp` | Parallela su CPU multicore, accumulatori privati per thread |
| CUDA naive | `src/kmeans_cuda.cu` | Traduzione diretta: 1 thread/punto, atomici globali |
| CUDA ottimizzata | `src/kmeans_cuda.cu` | Coalescing, shared memory, atomici privatizzati, riduzione warp-shuffle |

## Compilazione

Il progetto compila anche **senza GPU**: in quel caso vengono costruite solo le
versioni seriale e OpenMP, e la suite di test si limita a quelle.

```bash
# con CMake (rileva automaticamente OpenMP e CUDA)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# oppure con il Makefile
make          # solo CPU
make cuda     # build completa, richiede nvcc nel PATH
```

## Esecuzione

```bash
# suite di correttezza
./build/kmeans_test

# benchmark singolo
./build/kmeans_bench --n 1000000 --d 32 --k 64 --iter 50 --reps 5

# campagna completa + grafici
./scripts/run_benchmarks.sh build/kmeans_bench
python3 scripts/plot_results.py
```

Opzioni principali di `kmeans_bench`: `--n --d --k --iter --tol --reps
--threads --impl --dataset --csv` (`--help` per l'elenco completo).

Su Google Colab si può usare direttamente `notebooks/kmeans_colab.ipynb`, che
copre compilazione, test, benchmark, grafici e profiling con Nsight.

## Struttura

```
include/kmeans.h              interfacce comuni alle quattro implementazioni
src/common.cpp                dataset sintetici, k-means++, I/O, metriche (ARI)
src/kmeans_serial.cpp         baseline
src/kmeans_omp.cpp            versione multicore
src/kmeans_cuda.cu            kernel naive e ottimizzati + wrapper host
src/main.cpp                  driver di benchmark
tests/test_correctness.cpp    suite di validazione
scripts/run_benchmarks.sh     campagna di misure (scalabilità in N, K, D, thread)
scripts/plot_results.py       grafici di speedup
notebooks/kmeans_colab.ipynb  esecuzione e profiling su GPU Colab
report/report.md              relazione
```

## Metodologia di misura

Alcune scelte che rendono i numeri interpretabili, discusse in dettaglio nel
report:

- **Stesso punto di partenza per tutti.** L'inizializzazione k-means++ è
  calcolata una volta sola e passata identica alle quattro implementazioni,
  quindi i cluster sono allineati per indice e i risultati si confrontano
  direttamente, senza bisogno di riallineare le etichette.
- **Inizializzazione esclusa dal cronometro.** Ciò che si misura è il loop di
  Lloyd, cioè la parte effettivamente parallelizzata.
- **Warm-up scartato e mediana su più ripetizioni**, non la media: più robusta
  rispetto alle interferenze dello scheduler.
- **Due tempi distinti per le versioni GPU**: `compute` (soli kernel) e `total`
  (incluse allocazioni device e trasferimenti H2D/D2H). Il secondo è quello
  onesto da citare come speedup complessivo.
- **Baseline seriale ottimizzata** (`-O3`, layout cache-friendly, nessuna radice
  quadrata nel confronto delle distanze). Una baseline debole gonfierebbe lo
  speedup e renderebbe l'analisi priva di valore.
