// ===========================================================================
//  K-means parallelo su CPU multicore con OpenMP.
//
//  Stesso algoritmo della versione seriale: cambia solo *chi* esegue i cicli.
//  Le tre tecniche usate qui hanno un corrispettivo diretto nella versione
//  CUDA, e il confronto fra le due e' uno dei punti piu' interessanti del
//  report:
//
//    OpenMP                          CUDA
//    ------------------------------  -----------------------------------------
//    #pragma omp parallel for        griglia di thread, 1 thread per punto
//    reduction(+:sse)                riduzione warp-shuffle + 1 atomico/blocco
//    accumulatori privati per thread accumulatori privati in shared memory
//
//  La differenza di scala e' cio' che rende necessarie le ottimizzazioni CUDA:
//  con 4-16 thread la contesa sugli accumulatori e' trascurabile e basterebbe
//  anche un banale #pragma omp atomic; con decine di migliaia di thread
//  concorrenti la stessa scelta serializzerebbe l'intero kernel.
// ===========================================================================

#include "kmeans.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

double assign_pass_omp(const Dataset& ds, const std::vector<float>& C, int k,
                       std::vector<int>& labels) {
    const int n = ds.n, d = ds.d;
    double sse = 0.0;

    // schedule(static): il carico per punto e' costante (sempre k*d operazioni),
    // quindi una ripartizione statica a blocchi contigui e' ottimale -- niente
    // overhead di scheduling e ogni thread lavora su una porzione contigua di
    // memoria, il che preserva la localita' di cache.
#pragma omp parallel for schedule(static) reduction(+ : sse)
    for (int i = 0; i < n; ++i) {
        const float* xi = &ds.x[static_cast<size_t>(i) * d];
        float best = FLT_MAX;
        int best_k = 0;
        for (int c = 0; c < k; ++c) {
            const float dist = sqdist(xi, &C[static_cast<size_t>(c) * d], d);
            if (dist < best) { best = dist; best_k = c; }
        }
        labels[i] = best_k;
        sse += best;
    }
    return sse;
}

}  // namespace

KMeansResult kmeans_omp(const Dataset& ds, const KMeansParams& p,
                        const std::vector<float>& init_centroids,
                        int num_threads) {
    const int n = ds.n, d = ds.d, k = p.k;

#ifdef _OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
#else
    (void)num_threads;  // compilato senza -fopenmp: degrada alla versione seriale
#endif

    KMeansResult r;
    r.centroids = init_centroids;
    r.labels.assign(n, 0);

    std::vector<double> sums(static_cast<size_t>(k) * d);
    std::vector<int> counts(k);
    const float tol2 = p.tol * p.tol;

    const double t0 = now_seconds();

    for (int it = 0; it < p.max_iter; ++it) {
        // --- 1. assegnazione (parallela sui punti) ------------------------
        assign_pass_omp(ds, r.centroids, k, r.labels);

        // --- 2. aggiornamento con privatizzazione degli accumulatori ------
        //
        // Ogni thread accumula in buffer propri e li fonde nei buffer globali
        // una sola volta, in sezione critica. Il costo della sincronizzazione
        // passa cosi' da O(n) a O(num_threads) ingressi in sezione critica.
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);

#pragma omp parallel
        {
            std::vector<double> local_sums(static_cast<size_t>(k) * d, 0.0);
            std::vector<int> local_counts(k, 0);

#pragma omp for schedule(static) nowait
            for (int i = 0; i < n; ++i) {
                const int c = r.labels[i];
                local_counts[c]++;
                const float* xi = &ds.x[static_cast<size_t>(i) * d];
                double* s = &local_sums[static_cast<size_t>(c) * d];
                for (int j = 0; j < d; ++j) s[j] += xi[j];
            }

#pragma omp critical
            {
                for (size_t t = 0; t < local_sums.size(); ++t) sums[t] += local_sums[t];
                for (int c = 0; c < k; ++c) counts[c] += local_counts[c];
            }
        }

        // --- 3. nuovi centroidi e spostamento massimo ---------------------
        //
        // O(k*d): con k e d piccoli rispetto a n il costo e' trascurabile e
        // parallelizzarlo non ripaga l'overhead di apertura della regione.
        float shift = 0.0f;
        for (int c = 0; c < k; ++c) {
            if (counts[c] == 0) continue;  // cluster vuoto: centroide invariato
            float* cc = &r.centroids[static_cast<size_t>(c) * d];
            const double inv = 1.0 / counts[c];
            float moved = 0.0f;
            for (int j = 0; j < d; ++j) {
                const float nv = static_cast<float>(sums[static_cast<size_t>(c) * d + j] * inv);
                const float t = nv - cc[j];
                moved = fmaf(t, t, moved);
                cc[j] = nv;
            }
            shift = std::max(shift, moved);
        }

        r.iterations = it + 1;
        if (shift <= tol2) break;
    }

    r.inertia = assign_pass_omp(ds, r.centroids, k, r.labels);

    r.time_compute = now_seconds() - t0;
    r.time_total = r.time_compute;
    return r;
}
