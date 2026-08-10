// ===========================================================================
//  Advanced Computer Architecture - Progetto GPU
//  K-means clustering: interfacce comuni a tutte le implementazioni
//
//  Le quattro implementazioni (seriale, OpenMP, CUDA naive, CUDA ottimizzata)
//  espongono la stessa identica firma e ricevono gli stessi centroidi iniziali,
//  cosi' che il confronto dei risultati sia diretto e il calcolo dello speedup
//  misuri esclusivamente le differenze di parallelizzazione.
// ===========================================================================

#ifndef ACA_KMEANS_H
#define ACA_KMEANS_H

#include <cstddef>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
//  Strutture dati
// ---------------------------------------------------------------------------

// Dataset in layout "point-major" (row-major): x[i*d + j] e' la coordinata j
// del punto i. E' il layout naturale per la CPU, perche' le d coordinate di uno
// stesso punto sono contigue e vengono percorse insieme nel calcolo di una
// distanza: ogni cache line caricata viene sfruttata per intero.
//
// La versione CUDA ottimizzata usa invece il layout "feature-major"
// (x[j*n + i]), che rende coalescenti gli accessi dei thread di uno stesso
// warp; la trasposizione avviene una sola volta, prima del trasferimento H2D.
struct Dataset {
    int n = 0;                      // numero di punti
    int d = 0;                      // numero di dimensioni (feature)
    std::vector<float> x;           // n*d, row-major
    std::vector<int> true_labels;   // n, valorizzato solo per i dataset sintetici
};

struct KMeansParams {
    int k = 8;              // numero di cluster
    int max_iter = 100;     // tetto sulle iterazioni di Lloyd
    float tol = 1e-4f;      // convergenza: max spostamento di un centroide
    unsigned seed = 42;     // seme per generazione dati e inizializzazione
};

struct KMeansResult {
    std::vector<float> centroids;   // k*d, row-major
    std::vector<int> labels;        // n
    double inertia = 0.0;           // SSE finale (somma delle distanze^2 dal centroide)
    int iterations = 0;             // iterazioni effettivamente eseguite
    double time_total = 0.0;        // s, end-to-end: include alloc device e H2D/D2H
    double time_compute = 0.0;      // s, solo calcolo: esclude i trasferimenti
};

// ---------------------------------------------------------------------------
//  Implementazioni
//
//  init_centroids: k*d valori, prodotti una sola volta da kmeanspp_init() e
//  condivisi da tutte le versioni. L'inizializzazione e' deliberatamente fuori
//  dalla regione cronometrata: cio' che vogliamo misurare e' il loop di Lloyd,
//  che e' la parte che parallelizziamo.
// ---------------------------------------------------------------------------

KMeansResult kmeans_serial(const Dataset& ds, const KMeansParams& p,
                           const std::vector<float>& init_centroids);

KMeansResult kmeans_omp(const Dataset& ds, const KMeansParams& p,
                        const std::vector<float>& init_centroids,
                        int num_threads = 0);   // 0 = tutti i core disponibili

// Definite solo quando il progetto e' compilato con il toolkit CUDA.
KMeansResult kmeans_cuda_naive(const Dataset& ds, const KMeansParams& p,
                               const std::vector<float>& init_centroids);

KMeansResult kmeans_cuda_opt(const Dataset& ds, const KMeansParams& p,
                             const std::vector<float>& init_centroids);

// Stampa nome, capability e caratteristiche della GPU (per la sezione
// "setup sperimentale" del report). Ritorna false se nessuna GPU e' visibile.
bool cuda_print_device_info();

// ---------------------------------------------------------------------------
//  Generazione dati e I/O
// ---------------------------------------------------------------------------

// Blob gaussiani isotropi: k centri estratti uniformemente in [-10,10]^d e n
// punti distribuiti attorno ad essi con deviazione standard cluster_std.
// Con cluster_std piccolo i cluster sono ben separati e il clustering corretto
// e' univoco, il che rende il test di correttezza inattaccabile.
Dataset generate_blobs(int n, int d, int k, float cluster_std, unsigned seed);

Dataset load_csv(const std::string& path);
bool save_csv(const std::string& path, const Dataset& ds);

// ---------------------------------------------------------------------------
//  Utility condivise
// ---------------------------------------------------------------------------

double now_seconds();

// Inizializzazione k-means++ (Arthur & Vassilvitskii, 2007): il primo centroide
// e' un punto casuale, i successivi sono estratti con probabilita' proporzionale
// alla distanza^2 dal centroide gia' scelto piu' vicino. Deterministica a parita'
// di seme, quindi tutte le implementazioni partono dallo stesso identico stato.
std::vector<float> kmeanspp_init(const Dataset& ds, int k, unsigned seed);

// Distanza euclidea al quadrato fra due punti di dimensione d.
// Aritmetica in float e uso di fmaf() per replicare la semantica dei kernel CUDA
// e massimizzare le probabilita' che le etichette coincidano bit-a-bit.
float sqdist(const float* a, const float* b, int d);

// ---------------------------------------------------------------------------
//  Verifica di correttezza
// ---------------------------------------------------------------------------

struct Comparison {
    double max_centroid_abs_diff = 0.0;  // differenza assoluta massima sui centroidi
    double max_centroid_rel_diff = 0.0;  // idem, relativa alla scala del centroide
    double label_agreement = 0.0;        // frazione di punti con la stessa etichetta [0,1]
    double inertia_rel_diff = 0.0;       // |SSE_a - SSE_b| / SSE_a
    int iteration_diff = 0;              // differenza nel numero di iterazioni
};

// Confronta due risultati. Non serve alcun matching fra cluster: tutte le
// implementazioni partono dagli stessi centroidi iniziali, quindi l'indice di
// ciascun cluster e' allineato per costruzione.
Comparison compare_results(const KMeansResult& ref, const KMeansResult& got);

// Adjusted Rand Index fra due partizioni. Vale 1 per partizioni identiche e ~0
// per partizioni indipendenti; e' invariante rispetto alla permutazione delle
// etichette, quindi e' la metrica giusta per confrontare il clustering ottenuto
// con la ground truth dei blob sintetici.
double adjusted_rand_index(const std::vector<int>& a, const std::vector<int>& b);

#endif  // ACA_KMEANS_H
