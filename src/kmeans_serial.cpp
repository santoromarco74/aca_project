// ===========================================================================
//  K-means seriale (algoritmo di Lloyd) - baseline di riferimento.
//
//  Questa e' la versione che compare al denominatore di ogni speedup, quindi
//  NON e' volutamente inefficiente: usa il layout row-major (cache-friendly per
//  la CPU), evita la radice quadrata nel confronto delle distanze e accumula in
//  double per contenere l'errore numerico. Una baseline scritta male gonfia
//  artificialmente lo speedup e rende l'analisi priva di valore.
//
//  Struttura di una iterazione:
//    1. assegnazione   O(n*k*d)  <- domina il tempo totale
//    2. aggiornamento  O(n*d)
//    3. test di convergenza sullo spostamento massimo dei centroidi
// ===========================================================================

#include "kmeans.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace {

// Passata di assegnazione: per ogni punto trova il centroide piu' vicino e
// accumula l'SSE. Ritorna l'inerzia della partizione prodotta.
double assign_pass(const Dataset& ds, const std::vector<float>& C, int k,
                   std::vector<int>& labels) {
    const int n = ds.n, d = ds.d;
    double sse = 0.0;
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

KMeansResult kmeans_serial(const Dataset& ds, const KMeansParams& p,
                           const std::vector<float>& init_centroids) {
    const int n = ds.n, d = ds.d, k = p.k;

    KMeansResult r;
    r.centroids = init_centroids;
    r.labels.assign(n, 0);

    // Accumulatori in double: sommare fino a n valori float in float
    // introdurrebbe un errore che si propaga direttamente sui centroidi.
    std::vector<double> sums(static_cast<size_t>(k) * d);
    std::vector<int> counts(k);
    const float tol2 = p.tol * p.tol;

    const double t0 = now_seconds();

    for (int it = 0; it < p.max_iter; ++it) {
        // --- 1. assegnazione ---------------------------------------------
        assign_pass(ds, r.centroids, k, r.labels);

        // --- 2. aggiornamento --------------------------------------------
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        for (int i = 0; i < n; ++i) {
            const int c = r.labels[i];
            counts[c]++;
            const float* xi = &ds.x[static_cast<size_t>(i) * d];
            double* s = &sums[static_cast<size_t>(c) * d];
            for (int j = 0; j < d; ++j) s[j] += xi[j];
        }

        // --- 3. nuovi centroidi e spostamento massimo ---------------------
        float shift = 0.0f;
        for (int c = 0; c < k; ++c) {
            // Cluster vuoto: il centroide resta dov'e'. E' la politica piu'
            // semplice e, soprattutto, deterministica -- indispensabile perche'
            // le quattro implementazioni producano lo stesso risultato.
            if (counts[c] == 0) continue;
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

    // Passata finale: etichette e SSE riferiti ai centroidi definitivi, cosi'
    // che i tre output del risultato siano fra loro coerenti. Le altre
    // implementazioni eseguono la stessa passata nel proprio paradigma.
    r.inertia = assign_pass(ds, r.centroids, k, r.labels);

    r.time_compute = now_seconds() - t0;
    r.time_total = r.time_compute;  // nessun trasferimento di memoria da contare
    return r;
}
