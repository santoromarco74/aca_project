// ===========================================================================
//  Utility condivise: generazione dataset, inizializzazione k-means++,
//  I/O CSV, metriche di confronto.
//
//  Tutto cio' che sta in questo file e' identico per le quattro implementazioni
//  e viene eseguito fuori dalle regioni cronometrate.
// ===========================================================================

#include "kmeans.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>

double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

float sqdist(const float* a, const float* b, int d) {
    float s = 0.0f;
    for (int j = 0; j < d; ++j) {
        const float t = a[j] - b[j];
        s = fmaf(t, t, s);
    }
    return s;
}

// ---------------------------------------------------------------------------
//  Generazione dataset sintetici
// ---------------------------------------------------------------------------

Dataset generate_blobs(int n, int d, int k, float cluster_std, unsigned seed) {
    Dataset ds;
    ds.n = n;
    ds.d = d;
    ds.x.resize(static_cast<size_t>(n) * d);
    ds.true_labels.resize(n);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> box(-10.0f, 10.0f);
    std::normal_distribution<float> noise(0.0f, cluster_std);

    // Centri veri dei cluster: sono la ground truth con cui confrontare i
    // centroidi ricostruiti dall'algoritmo.
    std::vector<float> centers(static_cast<size_t>(k) * d);
    for (float& c : centers) c = box(rng);

    std::uniform_int_distribution<int> pick(0, k - 1);
    for (int i = 0; i < n; ++i) {
        const int c = pick(rng);
        ds.true_labels[i] = c;
        for (int j = 0; j < d; ++j) {
            ds.x[static_cast<size_t>(i) * d + j] =
                centers[static_cast<size_t>(c) * d + j] + noise(rng);
        }
    }
    return ds;
}

// ---------------------------------------------------------------------------
//  Inizializzazione k-means++
//
//  Costo O(k*n*d), pari a una singola iterazione di assegnazione di Lloyd.
//  E' condivisa da tutte le implementazioni ed esclusa dalle misure di tempo:
//  parallelizzarla e' possibile (le n distanze di ogni passo sono indipendenti)
//  ma esula dal confronto che ci interessa.
// ---------------------------------------------------------------------------

std::vector<float> kmeanspp_init(const Dataset& ds, int k, unsigned seed) {
    const int n = ds.n, d = ds.d;
    std::vector<float> C(static_cast<size_t>(k) * d, 0.0f);
    if (n == 0 || k == 0) return C;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> uni(0, n - 1);

    // Primo centroide: un punto qualsiasi del dataset.
    const int first = uni(rng);
    std::copy_n(&ds.x[static_cast<size_t>(first) * d], d, &C[0]);

    // best[i] = distanza^2 dal centroide gia' scelto piu' vicino, mantenuta
    // incrementalmente per evitare di riscandire tutti i centroidi ad ogni passo.
    std::vector<double> best(n);
    for (int i = 0; i < n; ++i)
        best[i] = sqdist(&ds.x[static_cast<size_t>(i) * d], &C[0], d);

    for (int c = 1; c < k; ++c) {
        const double total = std::accumulate(best.begin(), best.end(), 0.0);
        int chosen;
        if (!(total > 0.0)) {
            // Tutti i punti coincidono con un centroide gia' scelto (accade se
            // k > numero di punti distinti): si ripiega su una scelta uniforme.
            chosen = uni(rng);
        } else {
            std::uniform_real_distribution<double> ur(0.0, total);
            const double target = ur(rng);
            double acc = 0.0;
            chosen = n - 1;
            for (int i = 0; i < n; ++i) {
                acc += best[i];
                if (acc >= target) { chosen = i; break; }
            }
        }
        std::copy_n(&ds.x[static_cast<size_t>(chosen) * d], d,
                    &C[static_cast<size_t>(c) * d]);
        for (int i = 0; i < n; ++i) {
            best[i] = std::min<double>(
                best[i], sqdist(&ds.x[static_cast<size_t>(i) * d],
                                &C[static_cast<size_t>(c) * d], d));
        }
    }
    return C;
}

// ---------------------------------------------------------------------------
//  I/O CSV
// ---------------------------------------------------------------------------

Dataset load_csv(const std::string& path) {
    Dataset ds;
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "[errore] impossibile aprire %s\n", path.c_str());
        return ds;
    }
    std::string line;
    std::vector<float> row;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        row.clear();
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            if (cell.empty()) continue;
            row.push_back(std::strtof(cell.c_str(), nullptr));
        }
        if (row.empty()) continue;
        if (ds.d == 0) ds.d = static_cast<int>(row.size());
        if (static_cast<int>(row.size()) != ds.d) {
            std::fprintf(stderr, "[errore] riga con %zu colonne, attese %d\n",
                         row.size(), ds.d);
            return Dataset{};
        }
        ds.x.insert(ds.x.end(), row.begin(), row.end());
        ++ds.n;
    }
    return ds;
}

bool save_csv(const std::string& path, const Dataset& ds) {
    std::ofstream out(path);
    if (!out) return false;
    for (int i = 0; i < ds.n; ++i) {
        for (int j = 0; j < ds.d; ++j) {
            if (j) out << ',';
            out << ds.x[static_cast<size_t>(i) * ds.d + j];
        }
        out << '\n';
    }
    return true;
}

// ---------------------------------------------------------------------------
//  Confronto fra risultati
// ---------------------------------------------------------------------------

Comparison compare_results(const KMeansResult& ref, const KMeansResult& got) {
    Comparison cmp;
    const size_t nc = std::min(ref.centroids.size(), got.centroids.size());
    for (size_t i = 0; i < nc; ++i) {
        const double a = ref.centroids[i], b = got.centroids[i];
        const double abs_diff = std::fabs(a - b);
        cmp.max_centroid_abs_diff = std::max(cmp.max_centroid_abs_diff, abs_diff);
        const double scale = std::max(1e-6, std::fabs(a));
        cmp.max_centroid_rel_diff = std::max(cmp.max_centroid_rel_diff, abs_diff / scale);
    }

    const size_t nl = std::min(ref.labels.size(), got.labels.size());
    size_t same = 0;
    for (size_t i = 0; i < nl; ++i) same += (ref.labels[i] == got.labels[i]);
    cmp.label_agreement = nl ? static_cast<double>(same) / nl : 1.0;

    const double denom = std::max(1e-12, std::fabs(ref.inertia));
    cmp.inertia_rel_diff = std::fabs(ref.inertia - got.inertia) / denom;
    cmp.iteration_diff = got.iterations - ref.iterations;
    return cmp;
}

double adjusted_rand_index(const std::vector<int>& a, const std::vector<int>& b) {
    const size_t n = std::min(a.size(), b.size());
    if (n < 2) return 1.0;

    // Tabella di contingenza sparsa: (etichetta_a, etichetta_b) -> conteggio.
    std::map<std::pair<int, int>, long long> joint;
    std::map<int, long long> ca, cb;
    for (size_t i = 0; i < n; ++i) {
        joint[{a[i], b[i]}]++;
        ca[a[i]]++;
        cb[b[i]]++;
    }

    auto choose2 = [](long long v) { return static_cast<double>(v) * (v - 1) / 2.0; };

    double sum_ij = 0.0, sum_a = 0.0, sum_b = 0.0;
    for (const auto& kv : joint) sum_ij += choose2(kv.second);
    for (const auto& kv : ca) sum_a += choose2(kv.second);
    for (const auto& kv : cb) sum_b += choose2(kv.second);

    const double total = choose2(static_cast<long long>(n));
    const double expected = sum_a * sum_b / total;
    const double max_index = 0.5 * (sum_a + sum_b);
    const double denom = max_index - expected;
    if (std::fabs(denom) < 1e-12) return 1.0;
    return (sum_ij - expected) / denom;
}
