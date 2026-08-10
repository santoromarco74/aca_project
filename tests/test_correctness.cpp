// ===========================================================================
//  Suite di test di correttezza.
//
//  Copre i tre livelli richiesti dalla traccia ("correttezza del codice su
//  istanze di benchmark significative"):
//
//   A. VALIDITA' ASSOLUTA - su blob ben separati il clustering ottimo e' unico
//      e noto, quindi si verifica che la baseline seriale ricostruisca la
//      ground truth (Adjusted Rand Index ~ 1) e che i centroidi trovati
//      coincidano con i centri veri.
//
//   B. EQUIVALENZA FRA IMPLEMENTAZIONI - le versioni parallele devono produrre
//      lo stesso risultato della seriale. Non si pretende l'uguaglianza
//      bit-a-bit: l'ordine di somma cambia fra le implementazioni e la somma in
//      virgola mobile non e' associativa. Si richiede quindi accordo sulle
//      etichette >= 99.9%, differenza relativa sui centroidi < 1e-3 e sull'SSE
//      < 1e-4. Su dati ben separati l'accordo osservato e' del 100%.
//
//   C. CASI LIMITE - k=1, k=n, d=1, cluster destinati a restare vuoti,
//      dataset con punti duplicati.
// ===========================================================================

#include "kmeans.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (ok) {
        std::printf("  [ ok ] %s\n", what.c_str());
    } else {
        std::printf("  [FAIL] %s\n", what.c_str());
        ++g_failures;
    }
}

void section(const std::string& title) {
    std::printf("\n=== %s ===\n", title.c_str());
}

// Soglie di tolleranza, giustificate nel commento di intestazione.
constexpr double kMaxCentroidRelDiff = 1e-3;
constexpr double kMaxInertiaRelDiff = 1e-4;
constexpr double kMinLabelAgreement = 0.999;

void check_equivalence(const std::string& name, const KMeansResult& ref,
                       const KMeansResult& got) {
    const Comparison c = compare_results(ref, got);
    std::printf("  %-12s max|dC|/|C|=%.3e  dSSE/SSE=%.3e  labels=%.4f%%  iter %+d\n",
                name.c_str(), c.max_centroid_rel_diff, c.inertia_rel_diff,
                100.0 * c.label_agreement, c.iteration_diff);
    check(c.max_centroid_rel_diff < kMaxCentroidRelDiff, name + ": centroidi coerenti");
    check(c.inertia_rel_diff < kMaxInertiaRelDiff, name + ": SSE coerente");
    check(c.label_agreement >= kMinLabelAgreement, name + ": etichette coerenti");
}

// Esegue tutte le implementazioni disponibili su una configurazione e ne
// verifica l'equivalenza rispetto alla seriale.
void run_config(int n, int d, int k, float cluster_std, unsigned seed,
                int max_iter = 100) {
    std::printf("\n--- n=%d d=%d k=%d std=%.2f seed=%u ---\n", n, d, k, cluster_std, seed);
    const Dataset ds = generate_blobs(n, d, k, cluster_std, seed);

    KMeansParams p;
    p.k = k;
    p.max_iter = max_iter;
    p.tol = 1e-4f;
    p.seed = seed;

    const std::vector<float> init = kmeanspp_init(ds, k, seed);
    const KMeansResult ref = kmeans_serial(ds, p, init);

    check(ref.iterations > 0, "seriale: almeno una iterazione eseguita");
    check(std::isfinite(ref.inertia) && ref.inertia >= 0.0, "seriale: SSE finito e non negativo");

    check_equivalence("omp", ref, kmeans_omp(ds, p, init, 0));
#ifdef USE_CUDA
    check_equivalence("cuda_naive", ref, kmeans_cuda_naive(ds, p, init));
    check_equivalence("cuda_opt", ref, kmeans_cuda_opt(ds, p, init));
#endif
}

// L'SSE deve diminuire monotonicamente lungo le iterazioni: e' la proprieta'
// fondamentale dell'algoritmo di Lloyd e un ottimo indicatore di bug nella
// fase di aggiornamento (un errore negli accumulatori la viola subito).
void test_monotonicity() {
    section("Monotonia dell'SSE (proprieta' di Lloyd)");
    const Dataset ds = generate_blobs(20000, 8, 10, 2.0f, 7);
    const std::vector<float> init = kmeanspp_init(ds, 10, 7);

    double prev = -1.0;
    bool monotone = true;
    for (int iters = 1; iters <= 15; ++iters) {
        KMeansParams p;
        p.k = 10;
        p.max_iter = iters;
        p.tol = 0.0f;  // disattiva l'uscita anticipata: vogliamo esattamente `iters` passi
        p.seed = 7;
        const KMeansResult r = kmeans_serial(ds, p, init);
        if (prev >= 0.0 && r.inertia > prev * (1.0 + 1e-9)) monotone = false;
        prev = r.inertia;
    }
    check(monotone, "SSE non crescente al crescere delle iterazioni");
}

// Su blob ben separati il clustering corretto e' univoco: l'algoritmo deve
// ritrovare la partizione vera, a meno di una permutazione delle etichette
// (motivo per cui si usa l'ARI e non l'accuratezza).
void test_ground_truth() {
    section("A. Recupero della ground truth su blob ben separati");
    const Dataset ds = generate_blobs(50000, 4, 6, 0.35f, 123);
    KMeansParams p;
    p.k = 6;
    p.max_iter = 200;
    p.tol = 1e-5f;
    p.seed = 123;
    const std::vector<float> init = kmeanspp_init(ds, 6, 123);
    const KMeansResult r = kmeans_serial(ds, p, init);

    const double ari = adjusted_rand_index(ds.true_labels, r.labels);
    std::printf("  Adjusted Rand Index = %.6f\n", ari);
    check(ari > 0.99, "ARI > 0.99 rispetto alla ground truth");

    // Ogni cluster vero deve avere un centroide ricostruito molto vicino.
    double worst = 0.0;
    for (int c = 0; c < 6; ++c) {
        // centro vero del cluster c = media dei suoi punti
        std::vector<double> center(4, 0.0);
        long long cnt = 0;
        for (int i = 0; i < ds.n; ++i) {
            if (ds.true_labels[i] != c) continue;
            for (int j = 0; j < 4; ++j) center[j] += ds.x[static_cast<size_t>(i) * 4 + j];
            ++cnt;
        }
        if (cnt == 0) continue;
        for (double& v : center) v /= static_cast<double>(cnt);

        double best = 1e30;
        for (int q = 0; q < 6; ++q) {
            double dist = 0.0;
            for (int j = 0; j < 4; ++j) {
                const double t = center[j] - r.centroids[static_cast<size_t>(q) * 4 + j];
                dist += t * t;
            }
            best = std::min(best, std::sqrt(dist));
        }
        worst = std::max(worst, best);
    }
    std::printf("  distanza massima centro-vero <-> centroide piu' vicino = %.5f\n", worst);
    check(worst < 0.1, "tutti i centri veri sono ricostruiti entro 0.1");
}

void test_edge_cases() {
    section("C. Casi limite");

    {   // k = 1: il centroide deve essere la media di tutti i punti
        const Dataset ds = generate_blobs(5000, 3, 4, 1.0f, 11);
        KMeansParams p;
        p.k = 1;
        p.max_iter = 50;
        p.seed = 11;
        const std::vector<float> init = kmeanspp_init(ds, 1, 11);
        const KMeansResult r = kmeans_serial(ds, p, init);

        std::vector<double> mean(3, 0.0);
        for (int i = 0; i < ds.n; ++i)
            for (int j = 0; j < 3; ++j) mean[j] += ds.x[static_cast<size_t>(i) * 3 + j];
        double worst = 0.0;
        for (int j = 0; j < 3; ++j) {
            mean[j] /= ds.n;
            worst = std::max(worst, std::fabs(mean[j] - r.centroids[j]));
        }
        check(worst < 1e-3, "k=1: il centroide coincide con la media globale");
        check_equivalence("omp k=1", r, kmeans_omp(ds, p, init, 0));
#ifdef USE_CUDA
        check_equivalence("cuda_opt k=1", r, kmeans_cuda_opt(ds, p, init));
#endif
    }

    {   // k = n: ogni punto e' il proprio cluster, SSE deve essere ~0
        const int n = 200;
        const Dataset ds = generate_blobs(n, 2, 5, 1.0f, 13);
        KMeansParams p;
        p.k = n;
        p.max_iter = 50;
        p.seed = 13;
        const std::vector<float> init = kmeanspp_init(ds, n, 13);
        const KMeansResult r = kmeans_serial(ds, p, init);
        std::printf("  k=n: SSE = %.6e\n", r.inertia);
        check(r.inertia < 1e-2, "k=n: SSE praticamente nullo");
    }

    {   // d = 1: caso degenere sulla dimensionalita'
        run_config(10000, 1, 5, 0.5f, 17);
    }

    {   // punti tutti identici: tutti i cluster tranne uno restano vuoti.
        // Verifica la politica sui cluster vuoti (centroide invariato) e
        // l'assenza di divisioni per zero.
        Dataset ds;
        ds.n = 1000;
        ds.d = 4;
        ds.x.assign(static_cast<size_t>(ds.n) * ds.d, 3.5f);
        KMeansParams p;
        p.k = 8;
        p.max_iter = 20;
        p.seed = 19;
        const std::vector<float> init = kmeanspp_init(ds, 8, 19);
        const KMeansResult r = kmeans_serial(ds, p, init);
        check(std::isfinite(r.inertia), "punti duplicati: nessun NaN/Inf");
        check(r.inertia < 1e-3, "punti duplicati: SSE nullo");
        check_equivalence("omp duplicati", r, kmeans_omp(ds, p, init, 0));
#ifdef USE_CUDA
        check_equivalence("cuda_opt duplicati", r, kmeans_cuda_opt(ds, p, init));
#endif
    }
}

}  // namespace

int main() {
    std::printf("Suite di test - K-means (ACA GPU project)\n");
#ifdef USE_CUDA
    std::printf("Build con supporto CUDA: le versioni GPU sono incluse nei test.\n");
#else
    std::printf("Build senza CUDA: vengono testate solo seriale e OpenMP.\n");
#endif

    test_ground_truth();

    section("B. Equivalenza fra implementazioni");
    run_config(50000, 16, 8, 1.0f, 42);     // caso tipico
    run_config(100000, 32, 32, 2.0f, 99);   // k e d piu' grandi
    run_config(3000, 64, 4, 5.0f, 5);       // cluster sovrapposti, molte iterazioni
    run_config(1234, 7, 13, 1.5f, 3);       // dimensioni non multiple del blocco CUDA

    test_monotonicity();
    test_edge_cases();

    std::printf("\n%s\n", std::string(60, '=').c_str());
    std::printf("Verifiche eseguite: %d, fallite: %d\n", g_checks, g_failures);
    std::printf("%s\n", g_failures == 0 ? "TUTTI I TEST SUPERATI" : "TEST FALLITI");
    return g_failures == 0 ? 0 : 1;
}
