// ===========================================================================
//  K-means su GPU: due implementazioni a confronto.
//
//  ---------------------------------------------------------------------------
//  VERSIONE NAIVE  -- traduzione diretta del codice seriale
//  ---------------------------------------------------------------------------
//    * layout row-major ereditato dalla CPU;
//    * un thread per punto, nessun uso della shared memory;
//    * un atomicAdd sulla global memory per ogni singolo contributo;
//    * finalizzazione dei centroidi sull'host, con D2H + H2D ad ogni iterazione.
//  E' corretta e gia' molto piu' veloce del seriale, ma spreca gran parte della
//  banda di memoria e serializza sugli atomici.
//
//  ---------------------------------------------------------------------------
//  VERSIONE OTTIMIZZATA -- quattro interventi, tutti misurabili col profiler
//  ---------------------------------------------------------------------------
//   (1) LAYOUT FEATURE-MAJOR: X[j*N + i] invece di X[i*D + j].
//       Thread consecutivi leggono indirizzi consecutivi, quindi i 32 accessi
//       di un warp si fondono in poche transazioni da 128 byte anziche' in 32
//       transazioni sparse. E' l'intervento con l'impatto maggiore, perche' il
//       kernel di assegnazione e' limitato dalla banda di memoria.
//
//   (2) CENTROIDI IN SHARED MEMORY: ogni punto li rilegge K*D volte. Caricarli
//       una volta per blocco trasforma quelle letture da global (~centinaia di
//       cicli) a shared (~decine), e le rende indipendenti dalla cache L2.
//
//   (3) ACCUMULATORI PRIVATIZZATI: la fase di aggiornamento usa accumulatori in
//       shared memory privati del blocco; solo alla fine ciascun blocco versa i
//       propri parziali in global con un atomico per elemento. Il numero di
//       atomici globali scende da O(N*D) a O(numBlocchi*K*D).
//
//   (4) RIDUZIONE DELL'SSE CON WARP SHUFFLE + finalizzazione dei centroidi
//       direttamente sul device: per ogni iterazione si trasferisce all'host un
//       solo float (lo spostamento massimo) invece dell'intero array dei
//       centroidi. Elimina il ping-pong D2H/H2D del ciclo esterno.
// ===========================================================================

#include "kmeans.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        const cudaError_t err__ = (call);                                     \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "[CUDA] %s:%d %s -> %s\n", __FILE__,         \
                         __LINE__, #call, cudaGetErrorString(err__));         \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

namespace {

constexpr int kBlockSize = 256;   // 8 warp per blocco: buon compromesso fra
                                  // occupancy e shared memory disponibile
constexpr int kWarpSize = 32;

// ---------------------------------------------------------------------------
//  Primitive di riduzione
// ---------------------------------------------------------------------------

// Riduzione dentro un warp senza toccare la shared memory: i 32 lane si
// scambiano i valori direttamente via registro con __shfl_down_sync.
__inline__ __device__ double warp_reduce_sum(double v) {
    for (int off = kWarpSize / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// Riduzione sull'intero blocco: prima dentro ogni warp, poi fra i warp.
// shared[] deve avere spazio per blockDim.x/32 double (al massimo 32).
// Il valore ritornato e' significativo solo per il thread 0.
__inline__ __device__ double block_reduce_sum(double v, double* shared) {
    const int lane = threadIdx.x % kWarpSize;
    const int wid = threadIdx.x / kWarpSize;

    v = warp_reduce_sum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
    v = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}

// ===========================================================================
//  KERNEL - VERSIONE NAIVE
// ===========================================================================

// Un thread per punto. X e' row-major: i thread di un warp leggono indirizzi
// distanti D float l'uno dall'altro, quindi ogni accesso genera una transazione
// di memoria distinta -- la banda utile crolla a 1/D di quella disponibile.
__global__ void assign_naive_kernel(const float* X, const float* C, int* labels,
                                    double* sse, int N, int D, int K) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float best = FLT_MAX;
    int best_k = 0;
    for (int c = 0; c < K; ++c) {
        float dist = 0.0f;
        for (int j = 0; j < D; ++j) {
            const float t = X[static_cast<size_t>(i) * D + j] - C[c * D + j];
            dist = fmaf(t, t, dist);
        }
        if (dist < best) { best = dist; best_k = c; }
    }
    labels[i] = best_k;

    // Un atomico globale per ogni punto: tutti i thread della griglia si
    // contendono la stessa cella di memoria. E' esattamente cio' che la
    // versione ottimizzata elimina con la riduzione gerarchica.
    atomicAdd(sse, static_cast<double>(best));
}

// Accumulo dei centroidi con atomici globali: O(N*D) operazioni atomiche, tutte
// concentrate su sole K*D celle. Con K piccolo la contesa e' severissima.
__global__ void update_naive_kernel(const float* X, const int* labels,
                                    double* sums, int* counts, int N, int D, int K) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    (void)K;

    const int c = labels[i];
    atomicAdd(&counts[c], 1);
    for (int j = 0; j < D; ++j)
        atomicAdd(&sums[static_cast<size_t>(c) * D + j],
                  static_cast<double>(X[static_cast<size_t>(i) * D + j]));
}

// ===========================================================================
//  KERNEL - VERSIONE OTTIMIZZATA
// ===========================================================================

// Assegnazione con layout feature-major, centroidi in shared memory e
// riduzione gerarchica dell'SSE.
//
// Layout della shared memory (la base e' allineata a 16 byte, quindi i double
// vanno per primi per garantirne l'allineamento a 8):
//     [0 .. 31]                       32 double per la riduzione fra warp
//     [32*8 .. 32*8 + K*D*4)          K*D float con i centroidi (se ci stanno)
//
// use_shared e' uniforme su tutta la griglia: quando K*D e' troppo grande per
// la shared memory del device si ripiega sulla lettura da global, che resta
// comunque servita bene dalla cache read-only grazie a __restrict__.
__global__ void assign_opt_kernel(const float* __restrict__ X,
                                  const float* __restrict__ C,
                                  int* __restrict__ labels,
                                  double* __restrict__ sse,
                                  int N, int D, int K, bool use_shared) {
    extern __shared__ char smem_raw[];
    double* sred = reinterpret_cast<double*>(smem_raw);
    float* sC = reinterpret_cast<float*>(&sred[kWarpSize]);

    const float* Cp = C;
    if (use_shared) {
        // Caricamento cooperativo: i blockDim.x thread si spartiscono i K*D
        // valori con accessi contigui, quindi anche questa copia e' coalescente.
        for (int t = threadIdx.x; t < K * D; t += blockDim.x) sC[t] = C[t];
        __syncthreads();
        Cp = sC;
    }

    double local_sse = 0.0;

    // Grid-stride loop: il kernel resta corretto per qualunque dimensione della
    // griglia e nessun thread esce anticipatamente, condizione necessaria
    // perche' la riduzione a valle possa usare __syncthreads() e le shuffle.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * blockDim.x) {
        float best = FLT_MAX;
        int best_k = 0;
        for (int c = 0; c < K; ++c) {
            float dist = 0.0f;
            for (int j = 0; j < D; ++j) {
                // Accesso coalescente: al variare di i (thread adiacenti) gli
                // indirizzi sono consecutivi.
                const float t = X[static_cast<size_t>(j) * N + i] - Cp[c * D + j];
                dist = fmaf(t, t, dist);
            }
            if (dist < best) { best = dist; best_k = c; }
        }
        labels[i] = best_k;
        local_sse += best;
    }

    const double block_sse = block_reduce_sum(local_sse, sred);
    if (threadIdx.x == 0) atomicAdd(sse, block_sse);  // un solo atomico per blocco
}

// Aggiornamento con accumulatori privati del blocco in shared memory.
//
// Layout della shared memory:
//     [0 .. K*D)      float, somme parziali del blocco
//     [K*D .. K*D+K)  int,   conteggi parziali del blocco
//
// Le somme parziali sono in float (gli atomici float in shared sono nativi e
// veloci) mentre l'accumulatore globale e' in double: poiche' ogni blocco
// elabora solo blockDim.x punti, l'errore di somma in float resta confinato,
// e la somma fra blocchi -- quella con molti addendi -- avviene in doppia
// precisione. Costa quanto la versione tutta-float ed e' molto piu' accurata.
__global__ void update_opt_kernel(const float* __restrict__ X,
                                  const int* __restrict__ labels,
                                  double* __restrict__ sums,
                                  int* __restrict__ counts,
                                  int N, int D, int K, bool use_shared) {
    extern __shared__ char smem_raw[];
    float* sSums = reinterpret_cast<float*>(smem_raw);
    int* sCnt = reinterpret_cast<int*>(&sSums[K * D]);

    if (use_shared) {
        for (int t = threadIdx.x; t < K * D; t += blockDim.x) sSums[t] = 0.0f;
        for (int t = threadIdx.x; t < K; t += blockDim.x) sCnt[t] = 0;
        __syncthreads();
    }

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += gridDim.x * blockDim.x) {
        const int c = labels[i];
        if (use_shared) {
            atomicAdd(&sCnt[c], 1);
            for (int j = 0; j < D; ++j)
                atomicAdd(&sSums[c * D + j], X[static_cast<size_t>(j) * N + i]);
        } else {
            atomicAdd(&counts[c], 1);
            for (int j = 0; j < D; ++j)
                atomicAdd(&sums[static_cast<size_t>(c) * D + j],
                          static_cast<double>(X[static_cast<size_t>(j) * N + i]));
        }
    }

    if (use_shared) {
        __syncthreads();
        // Il test != 0 evita l'atomico globale per i cluster che questo blocco
        // non ha mai toccato: con K grande la maggior parte dei blocchi ne tocca
        // solo una frazione.
        for (int t = threadIdx.x; t < K * D; t += blockDim.x)
            if (sSums[t] != 0.0f) atomicAdd(&sums[t], static_cast<double>(sSums[t]));
        for (int t = threadIdx.x; t < K; t += blockDim.x)
            if (sCnt[t] != 0) atomicAdd(&counts[t], sCnt[t]);
    }
}

// Finalizzazione sul device: divide le somme per i conteggi, sovrascrive i
// centroidi e calcola lo spostamento massimo. Si lancia con un blocco solo
// (il lavoro e' O(K*D), trascurabile): all'host serve poi copiare un unico
// float invece dell'intero array dei centroidi.
__global__ void finalize_opt_kernel(float* __restrict__ C,
                                    const double* __restrict__ sums,
                                    const int* __restrict__ counts,
                                    float* __restrict__ shift_out, int D, int K) {
    extern __shared__ char smem_raw[];
    float* sred = reinterpret_cast<float*>(smem_raw);  // 32 float

    float local_max = 0.0f;
    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        const int cnt = counts[c];
        if (cnt == 0) continue;  // cluster vuoto: centroide invariato
        const double inv = 1.0 / static_cast<double>(cnt);
        float moved = 0.0f;
        for (int j = 0; j < D; ++j) {
            const size_t idx = static_cast<size_t>(c) * D + j;
            const float nv = static_cast<float>(sums[idx] * inv);
            const float t = nv - C[idx];
            moved = fmaf(t, t, moved);
            C[idx] = nv;
        }
        local_max = fmaxf(local_max, moved);
    }

    for (int off = kWarpSize / 2; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, off));

    const int lane = threadIdx.x % kWarpSize;
    const int wid = threadIdx.x / kWarpSize;
    if (lane == 0) sred[wid] = local_max;
    __syncthreads();

    if (threadIdx.x == 0) {
        const int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
        float m = 0.0f;
        for (int w = 0; w < num_warps; ++w) m = fmaxf(m, sred[w]);
        *shift_out = m;
    }
}

}  // namespace

// ===========================================================================
//  Host: informazioni sul device
// ===========================================================================

bool cuda_print_device_info() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::fprintf(stderr, "[CUDA] nessun device disponibile\n");
        return false;
    }
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("GPU              : %s (compute capability %d.%d)\n", prop.name,
                prop.major, prop.minor);
    std::printf("  SM             : %d, %d thread/SM max\n", prop.multiProcessorCount,
                prop.maxThreadsPerMultiProcessor);
    std::printf("  Shared mem     : %zu B/blocco\n", prop.sharedMemPerBlock);
    std::printf("  Memoria globale: %.2f GiB, bus %d bit @ %d MHz\n",
                prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
                prop.memoryBusWidth, prop.memoryClockRate / 1000);
    const double peak_bw = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;
    std::printf("  Banda di picco : %.1f GB/s\n", peak_bw);
    return true;
}

// ===========================================================================
//  Host: versione NAIVE
// ===========================================================================

KMeansResult kmeans_cuda_naive(const Dataset& ds, const KMeansParams& p,
                               const std::vector<float>& init_centroids) {
    const int n = ds.n, d = ds.d, k = p.k;
    const size_t bytes_x = static_cast<size_t>(n) * d * sizeof(float);
    const size_t bytes_c = static_cast<size_t>(k) * d * sizeof(float);

    KMeansResult r;
    r.centroids = init_centroids;
    r.labels.assign(n, 0);

    const double t_start = now_seconds();

    float *dX = nullptr, *dC = nullptr;
    int *dLabels = nullptr, *dCounts = nullptr;
    double *dSums = nullptr, *dSse = nullptr;
    CUDA_CHECK(cudaMalloc(&dX, bytes_x));
    CUDA_CHECK(cudaMalloc(&dC, bytes_c));
    CUDA_CHECK(cudaMalloc(&dLabels, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dCounts, static_cast<size_t>(k) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dSums, static_cast<size_t>(k) * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dSse, sizeof(double)));

    // I dati salgono sul device una sola volta, fuori dal ciclo iterativo:
    // e' la ragione per cui il costo dei trasferimenti si ammortizza bene.
    CUDA_CHECK(cudaMemcpy(dX, ds.x.data(), bytes_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, r.centroids.data(), bytes_c, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    const int grid = (n + kBlockSize - 1) / kBlockSize;
    const float tol2 = p.tol * p.tol;
    std::vector<double> sums(static_cast<size_t>(k) * d);
    std::vector<int> counts(k);

    const double t_compute0 = now_seconds();

    for (int it = 0; it < p.max_iter; ++it) {
        CUDA_CHECK(cudaMemset(dSse, 0, sizeof(double)));
        assign_naive_kernel<<<grid, kBlockSize>>>(dX, dC, dLabels, dSse, n, d, k);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemset(dSums, 0, static_cast<size_t>(k) * d * sizeof(double)));
        CUDA_CHECK(cudaMemset(dCounts, 0, static_cast<size_t>(k) * sizeof(int)));
        update_naive_kernel<<<grid, kBlockSize>>>(dX, dLabels, dSums, dCounts, n, d, k);
        CUDA_CHECK(cudaGetLastError());

        // Finalizzazione sull'host: costringe a un D2H dei parziali e a un H2D
        // dei nuovi centroidi ad ogni iterazione, con sincronizzazione completa.
        CUDA_CHECK(cudaMemcpy(sums.data(), dSums,
                              static_cast<size_t>(k) * d * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(counts.data(), dCounts,
                              static_cast<size_t>(k) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        float shift = 0.0f;
        for (int c = 0; c < k; ++c) {
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
        CUDA_CHECK(cudaMemcpy(dC, r.centroids.data(), bytes_c, cudaMemcpyHostToDevice));

        r.iterations = it + 1;
        if (shift <= tol2) break;
    }

    // Passata finale con i centroidi definitivi.
    CUDA_CHECK(cudaMemset(dSse, 0, sizeof(double)));
    assign_naive_kernel<<<grid, kBlockSize>>>(dX, dC, dLabels, dSse, n, d, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    r.time_compute = now_seconds() - t_compute0;

    CUDA_CHECK(cudaMemcpy(&r.inertia, dSse, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.labels.data(), dLabels,
                          static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dLabels));
    CUDA_CHECK(cudaFree(dCounts));
    CUDA_CHECK(cudaFree(dSums));
    CUDA_CHECK(cudaFree(dSse));

    r.time_total = now_seconds() - t_start;
    return r;
}

// ===========================================================================
//  Host: versione OTTIMIZZATA
// ===========================================================================

KMeansResult kmeans_cuda_opt(const Dataset& ds, const KMeansParams& p,
                             const std::vector<float>& init_centroids) {
    const int n = ds.n, d = ds.d, k = p.k;
    const size_t bytes_x = static_cast<size_t>(n) * d * sizeof(float);
    const size_t bytes_c = static_cast<size_t>(k) * d * sizeof(float);

    KMeansResult r;
    r.centroids = init_centroids;
    r.labels.assign(n, 0);

    const double t_start = now_seconds();

    // Trasposizione row-major -> feature-major. Costo O(n*d) pagato una sola
    // volta, contro un guadagno che si ripete ad ogni iterazione di Lloyd.
    std::vector<float> xt(static_cast<size_t>(n) * d);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < d; ++j)
            xt[static_cast<size_t>(j) * n + i] = ds.x[static_cast<size_t>(i) * d + j];

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    const size_t smem_limit = prop.sharedMemPerBlock;

    float *dX = nullptr, *dC = nullptr, *dShift = nullptr;
    int *dLabels = nullptr, *dCounts = nullptr;
    double *dSums = nullptr, *dSse = nullptr;
    CUDA_CHECK(cudaMalloc(&dX, bytes_x));
    CUDA_CHECK(cudaMalloc(&dC, bytes_c));
    CUDA_CHECK(cudaMalloc(&dShift, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dLabels, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dCounts, static_cast<size_t>(k) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dSums, static_cast<size_t>(k) * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dSse, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(dX, xt.data(), bytes_x, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, r.centroids.data(), bytes_c, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Dimensionamento della shared memory, con ripiego su global se K*D non ci sta.
    const size_t smem_assign_full =
        kWarpSize * sizeof(double) + static_cast<size_t>(k) * d * sizeof(float);
    const bool assign_shared = smem_assign_full <= smem_limit;
    const size_t smem_assign =
        assign_shared ? smem_assign_full : kWarpSize * sizeof(double);

    const size_t smem_update_full = static_cast<size_t>(k) * d * sizeof(float) +
                                    static_cast<size_t>(k) * sizeof(int);
    const bool update_shared = smem_update_full <= smem_limit;
    const size_t smem_update = update_shared ? smem_update_full : 0;

    const int grid = (n + kBlockSize - 1) / kBlockSize;
    const float tol2 = p.tol * p.tol;

    const double t_compute0 = now_seconds();

    for (int it = 0; it < p.max_iter; ++it) {
        CUDA_CHECK(cudaMemsetAsync(dSse, 0, sizeof(double)));
        assign_opt_kernel<<<grid, kBlockSize, smem_assign>>>(dX, dC, dLabels, dSse,
                                                             n, d, k, assign_shared);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemsetAsync(dSums, 0, static_cast<size_t>(k) * d * sizeof(double)));
        CUDA_CHECK(cudaMemsetAsync(dCounts, 0, static_cast<size_t>(k) * sizeof(int)));
        update_opt_kernel<<<grid, kBlockSize, smem_update>>>(dX, dLabels, dSums, dCounts,
                                                             n, d, k, update_shared);
        CUDA_CHECK(cudaGetLastError());

        finalize_opt_kernel<<<1, kBlockSize, kWarpSize * sizeof(float)>>>(
            dC, dSums, dCounts, dShift, d, k);
        CUDA_CHECK(cudaGetLastError());

        // Unico trasferimento del ciclo: 4 byte.
        float shift = 0.0f;
        CUDA_CHECK(cudaMemcpy(&shift, dShift, sizeof(float), cudaMemcpyDeviceToHost));

        r.iterations = it + 1;
        if (shift <= tol2) break;
    }

    CUDA_CHECK(cudaMemsetAsync(dSse, 0, sizeof(double)));
    assign_opt_kernel<<<grid, kBlockSize, smem_assign>>>(dX, dC, dLabels, dSse,
                                                         n, d, k, assign_shared);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    r.time_compute = now_seconds() - t_compute0;

    CUDA_CHECK(cudaMemcpy(&r.inertia, dSse, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.labels.data(), dLabels,
                          static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.centroids.data(), dC, bytes_c, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dShift));
    CUDA_CHECK(cudaFree(dLabels));
    CUDA_CHECK(cudaFree(dCounts));
    CUDA_CHECK(cudaFree(dSums));
    CUDA_CHECK(cudaFree(dSse));

    r.time_total = now_seconds() - t_start;
    return r;
}
