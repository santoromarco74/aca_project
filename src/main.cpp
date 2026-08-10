// ===========================================================================
//  Driver di benchmark.
//
//  Esegue le implementazioni richieste sullo stesso dataset e con gli stessi
//  centroidi iniziali, ripete ogni misura piu' volte scartando la prima
//  (warm-up) e produce una tabella a schermo piu' un CSV per i grafici.
//
//  Esempi d'uso:
//    ./kmeans_bench --n 200000 --d 32 --k 16 --reps 5
//    ./kmeans_bench --n 1000000 --d 64 --k 64 --impl serial,omp,cuda_opt
//    ./kmeans_bench --dataset data/iris.csv --k 3 --impl all --csv out.csv
// ===========================================================================

#include "kmeans.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct Options {
    int n = 200000;
    int d = 32;
    int k = 16;
    int max_iter = 100;
    float tol = 1e-4f;
    float cluster_std = 1.0f;
    unsigned seed = 42;
    int reps = 3;              // ripetizioni cronometrate, oltre al warm-up
    int threads = 0;           // 0 = tutti i core
    std::string impls = "all";
    std::string dataset;       // se non vuoto, carica da CSV invece di generare
    std::string csv;           // file di output per i grafici
};

void print_usage(const char* argv0) {
    std::printf(
        "Uso: %s [opzioni]\n"
        "  --n N            numero di punti del dataset sintetico (default 200000)\n"
        "  --d D            dimensionalita' (default 32)\n"
        "  --k K            numero di cluster (default 16)\n"
        "  --iter I         massimo numero di iterazioni di Lloyd (default 100)\n"
        "  --tol T          soglia di convergenza sui centroidi (default 1e-4)\n"
        "  --std S          deviazione standard dei blob sintetici (default 1.0)\n"
        "  --seed S         seme del generatore (default 42)\n"
        "  --reps R         ripetizioni cronometrate per implementazione (default 3)\n"
        "  --threads T      thread OpenMP, 0 = tutti i core (default 0)\n"
        "  --impl LIST      serial,omp,cuda_naive,cuda_opt oppure all (default all)\n"
        "  --dataset FILE   carica il dataset da CSV invece di generarlo\n"
        "  --csv FILE       accoda i risultati grezzi a un CSV\n"
        "  --help           mostra questo messaggio\n",
        argv0);
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "[errore] %s richiede un valore\n", name);
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };
        if (a == "--n") o.n = std::atoi(next("--n"));
        else if (a == "--d") o.d = std::atoi(next("--d"));
        else if (a == "--k") o.k = std::atoi(next("--k"));
        else if (a == "--iter") o.max_iter = std::atoi(next("--iter"));
        else if (a == "--tol") o.tol = std::atof(next("--tol"));
        else if (a == "--std") o.cluster_std = std::atof(next("--std"));
        else if (a == "--seed") o.seed = std::strtoul(next("--seed"), nullptr, 10);
        else if (a == "--reps") o.reps = std::atoi(next("--reps"));
        else if (a == "--threads") o.threads = std::atoi(next("--threads"));
        else if (a == "--impl") o.impls = next("--impl");
        else if (a == "--dataset") o.dataset = next("--dataset");
        else if (a == "--csv") o.csv = next("--csv");
        else if (a == "--help" || a == "-h") { print_usage(argv[0]); return false; }
        else {
            std::fprintf(stderr, "[errore] opzione sconosciuta: %s\n", a.c_str());
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    return true;
}

bool wants(const std::string& list, const std::string& name) {
    if (list == "all") return true;
    return list.find(name) != std::string::npos;
}

struct Timing {
    std::string name;
    KMeansResult best;         // risultato della ripetizione piu' veloce
    double median_total = 0.0;
    double median_compute = 0.0;
    bool valid = false;
};

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return (v.size() % 2) ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// Esegue reps+1 volte la funzione fornita: la prima esecuzione e' un warm-up
// (allocazioni lazy, popolamento delle cache, creazione del contesto CUDA) e
// viene scartata. Delle restanti si prende la mediana, piu' robusta della media
// rispetto agli outlier dovuti allo scheduler del sistema operativo.
template <typename F>
Timing run_benchmark(const std::string& name, int reps, F&& fn) {
    Timing t;
    t.name = name;
    std::vector<double> tot, comp;
    for (int rep = 0; rep < reps + 1; ++rep) {
        KMeansResult r = fn();
        if (rep == 0) { t.best = r; continue; }   // warm-up scartato
        tot.push_back(r.time_total);
        comp.push_back(r.time_compute);
        if (r.time_compute < t.best.time_compute || t.best.labels.empty()) t.best = r;
    }
    t.median_total = median(tot);
    t.median_compute = median(comp);
    t.valid = true;
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse_args(argc, argv, o)) return 0;

    // --- dataset -----------------------------------------------------------
    Dataset ds;
    if (!o.dataset.empty()) {
        ds = load_csv(o.dataset);
        if (ds.n == 0) { std::fprintf(stderr, "[errore] dataset vuoto\n"); return 1; }
        o.n = ds.n;
        o.d = ds.d;
        std::printf("Dataset   : %s (n=%d, d=%d)\n", o.dataset.c_str(), ds.n, ds.d);
    } else {
        ds = generate_blobs(o.n, o.d, o.k, o.cluster_std, o.seed);
        std::printf("Dataset   : blob sintetici (n=%d, d=%d, k=%d, std=%.2f, seed=%u)\n",
                    o.n, o.d, o.k, o.cluster_std, o.seed);
    }
    std::printf("Parametri : k=%d, max_iter=%d, tol=%.1e, reps=%d (+1 warm-up)\n",
                o.k, o.max_iter, o.tol, o.reps);
#ifdef _OPENMP
    std::printf("OpenMP    : %d thread disponibili\n",
                o.threads > 0 ? o.threads : omp_get_max_threads());
#else
    std::printf("OpenMP    : non disponibile (compilato senza -fopenmp)\n");
#endif
#ifdef USE_CUDA
    cuda_print_device_info();
#else
    std::printf("CUDA      : non disponibile (compilato senza toolkit CUDA)\n");
#endif

    KMeansParams p;
    p.k = o.k;
    p.max_iter = o.max_iter;
    p.tol = o.tol;
    p.seed = o.seed;

    // Centroidi iniziali comuni a tutte le implementazioni: e' cio' che rende
    // il confronto dei risultati diretto, senza bisogno di riallineare i cluster.
    const double t_init = now_seconds();
    const std::vector<float> init = kmeanspp_init(ds, o.k, o.seed);
    std::printf("Init      : k-means++ in %.3f s (esclusa dalle misure)\n\n",
                now_seconds() - t_init);

    // --- esecuzione --------------------------------------------------------
    std::vector<Timing> results;

    if (wants(o.impls, "serial"))
        results.push_back(run_benchmark("serial", o.reps,
                                        [&] { return kmeans_serial(ds, p, init); }));
    if (wants(o.impls, "omp"))
        results.push_back(run_benchmark("omp", o.reps,
                                        [&] { return kmeans_omp(ds, p, init, o.threads); }));
#ifdef USE_CUDA
    if (wants(o.impls, "cuda_naive"))
        results.push_back(run_benchmark("cuda_naive", o.reps,
                                        [&] { return kmeans_cuda_naive(ds, p, init); }));
    if (wants(o.impls, "cuda_opt"))
        results.push_back(run_benchmark("cuda_opt", o.reps,
                                        [&] { return kmeans_cuda_opt(ds, p, init); }));
#endif

    if (results.empty()) {
        std::fprintf(stderr, "[errore] nessuna implementazione selezionata\n");
        return 1;
    }

    // --- tabella dei tempi -------------------------------------------------
    const double base = results.front().median_compute;

    std::printf("%-12s %10s %10s %9s %9s %6s %14s\n", "impl", "compute(s)",
                "total(s)", "sp.comp", "sp.tot", "iter", "SSE");
    std::printf("%s\n", std::string(76, '-').c_str());
    for (const Timing& t : results) {
        std::printf("%-12s %10.4f %10.4f %9.2f %9.2f %6d %14.4f\n", t.name.c_str(),
                    t.median_compute, t.median_total,
                    t.median_compute > 0 ? base / t.median_compute : 0.0,
                    t.median_total > 0 ? results.front().median_total / t.median_total : 0.0,
                    t.best.iterations, t.best.inertia);
    }

    // --- correttezza rispetto alla baseline --------------------------------
    if (results.size() > 1) {
        std::printf("\nCorrettezza rispetto a '%s':\n", results.front().name.c_str());
        std::printf("%-12s %14s %14s %14s %8s\n", "impl", "max|dC|", "max|dC|/|C|",
                    "dSSE/SSE", "labels");
        std::printf("%s\n", std::string(66, '-').c_str());
        for (size_t i = 1; i < results.size(); ++i) {
            const Comparison c = compare_results(results.front().best, results[i].best);
            std::printf("%-12s %14.3e %14.3e %14.3e %7.3f%%\n", results[i].name.c_str(),
                        c.max_centroid_abs_diff, c.max_centroid_rel_diff,
                        c.inertia_rel_diff, 100.0 * c.label_agreement);
        }
    }

    // --- qualita' del clustering rispetto alla ground truth ----------------
    if (!ds.true_labels.empty()) {
        const double ari = adjusted_rand_index(ds.true_labels, results.front().best.labels);
        std::printf("\nAdjusted Rand Index vs ground truth: %.4f\n", ari);
    }

    // --- CSV ---------------------------------------------------------------
    if (!o.csv.empty()) {
        const bool exists = std::ifstream(o.csv).good();
        std::ofstream out(o.csv, std::ios::app);
        if (!out) {
            std::fprintf(stderr, "[errore] impossibile scrivere %s\n", o.csv.c_str());
            return 1;
        }
        if (!exists)
            out << "impl,n,d,k,threads,iterations,inertia,time_compute_s,time_total_s\n";
        int threads = 1;
#ifdef _OPENMP
        threads = o.threads > 0 ? o.threads : omp_get_max_threads();
#endif
        for (const Timing& t : results) {
            out << t.name << ',' << o.n << ',' << o.d << ',' << o.k << ','
                << (t.name == "omp" ? threads : 1) << ',' << t.best.iterations << ','
                << t.best.inertia << ',' << t.median_compute << ',' << t.median_total << '\n';
        }
        std::printf("\nRisultati accodati a %s\n", o.csv.c_str());
    }

    return 0;
}
