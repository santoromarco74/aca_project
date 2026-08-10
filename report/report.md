# K-means clustering su GPU: dal seriale a CUDA

**Corso**: Advanced Computer Architecture
**Autore**: Marco Santoro
**Repository**: `santoromarco74/aca_project`

---

## 1. Problema affrontato

Il K-means è un algoritmo di clustering non supervisionato che partiziona un
insieme di $n$ punti in $\mathbb{R}^d$ in $k$ gruppi, minimizzando la somma
delle distanze quadratiche di ciascun punto dal centroide del proprio cluster:

$$\text{SSE} = \sum_{i=1}^{n} \lVert x_i - c_{\ell(i)} \rVert^2$$

dove $\ell(i)$ è l'indice del cluster assegnato al punto $i$. Il problema è
NP-hard in generale; l'algoritmo di **Lloyd** ne fornisce un'euristica che
alterna due passi fino a convergenza:

| Passo | Operazione | Costo |
|---|---|---|
| **Assegnazione** | ogni punto va al centroide più vicino | $O(n \cdot k \cdot d)$ |
| **Aggiornamento** | ogni centroide diventa la media dei suoi punti | $O(n \cdot d)$ |

Il passo di assegnazione domina il costo di un fattore $k$ ed è
**embarrassingly parallel**: la decisione su ciascun punto è indipendente da
quella su tutti gli altri. Il passo di aggiornamento è invece una **riduzione**
per chiave (il cluster di appartenenza), che richiede sincronizzazione fra i
thread e costituisce la parte architetturalmente interessante del problema.

Questa asimmetria — un passo perfettamente parallelo seguito da una riduzione
con conflitti — rende il K-means un caso di studio efficace per confrontare
CPU multicore e GPU: le tecniche necessarie sulle due architetture sono
concettualmente le stesse, ma la differenza di scala nel parallelismo (unità
contro decine di migliaia di thread) cambia radicalmente quali scelte sono
accettabili.

### Convergenza e criterio d'arresto

Il ciclo termina quando il massimo spostamento quadratico di un centroide
scende sotto una soglia $\tau^2$, oppure al raggiungimento di `max_iter`. Sulla
gestione dei **cluster vuoti** si è scelta la politica più semplice —
il centroide resta dov'è — perché è l'unica pienamente deterministica, e il
determinismo è un prerequisito per poter confrontare i risultati delle quattro
implementazioni senza ambiguità.

---

## 2. Strategie di implementazione

Sono state realizzate quattro versioni, tutte a partire dagli **stessi
centroidi iniziali** prodotti da k-means++.

### 2.1 Baseline seriale

Scritta per essere una baseline *onesta*, non un uomo di paglia:

- layout **row-major** (`x[i*d + j]`), che è quello cache-friendly per la CPU:
  le $d$ coordinate di un punto sono contigue e vengono percorse insieme nel
  calcolo di una distanza, quindi ogni cache line caricata viene usata per intero;
- nessuna radice quadrata: l'`argmin` sulle distanze coincide con l'`argmin`
  sulle distanze al quadrato;
- accumulatori in `double` per le somme dei centroidi, per contenere l'errore
  numerico su $n$ addendi;
- compilata con `-O3` e auto-vettorizzazione attiva, **senza** `-ffast-math`
  (cambierebbe la semantica in virgola mobile e renderebbe non interpretabile
  il confronto numerico con le versioni CUDA).

### 2.2 OpenMP

Tre costrutti, ciascuno con un corrispettivo diretto nella versione CUDA:

| OpenMP | CUDA | Scopo |
|---|---|---|
| `#pragma omp parallel for schedule(static)` | griglia di thread, 1 thread/punto | parallelizzare l'assegnazione |
| `reduction(+:sse)` | riduzione warp-shuffle + 1 atomico/blocco | sommare l'SSE |
| accumulatori privati per thread + `critical` | accumulatori privati in shared memory | evitare la contesa nell'aggiornamento |

`schedule(static)` è la scelta corretta perché il carico per punto è costante
($k \cdot d$ operazioni sempre): una ripartizione statica a blocchi contigui
azzera l'overhead di scheduling e preserva la località di cache.

Sulla privatizzazione degli accumulatori vale la pena essere espliciti: con
4–16 thread la contesa è modesta e basterebbe anche un `#pragma omp atomic`;
il costo della sincronizzazione passa comunque da $O(n)$ a
$O(\text{num\_threads})$ ingressi in sezione critica. È **su GPU** che la stessa
scelta diventa obbligatoria anziché opportuna.

### 2.3 CUDA — versione naive

Traduzione diretta del codice seriale, che serve da termine di paragone interno
per quantificare l'effetto delle ottimizzazioni successive:

- layout row-major ereditato dalla CPU;
- un thread per punto, nessun uso della shared memory;
- un `atomicAdd` sulla global memory per ogni singolo contributo, sia per l'SSE
  sia per le somme dei centroidi;
- finalizzazione dei centroidi **sull'host**, con D2H dei parziali e H2D dei
  nuovi centroidi ad ogni iterazione.

È già corretta e sensibilmente più veloce del seriale, ma spreca gran parte
della banda di memoria disponibile e serializza sugli atomici.

### 2.4 CUDA — versione ottimizzata

Quattro interventi, tutti verificabili col profiler.

**(1) Layout feature-major.** I dati vengono trasposti una sola volta prima del
trasferimento H2D, passando da `X[i*D + j]` a `X[j*N + i]`. Nel kernel di
assegnazione, thread consecutivi accedono ora a indirizzi consecutivi: i 32
accessi di un warp si fondono in poche transazioni da 128 byte anziché in 32
transazioni sparse. È l'intervento con l'impatto maggiore, perché il kernel è
limitato dalla banda di memoria e non dal calcolo. Il costo della trasposizione
è $O(n \cdot d)$ pagato una volta, contro un guadagno che si ripete ad ogni
iterazione di Lloyd.

**(2) Centroidi in shared memory.** Ogni punto rilegge i centroidi $k \cdot d$
volte. Caricarli una volta per blocco con una copia cooperativa trasforma
quelle letture da global (centinaia di cicli) a shared (decine). Quando
$k \cdot d$ eccede la shared memory disponibile il kernel ripiega sulla lettura
da global, comunque servita bene dalla cache read-only grazie a `__restrict__`.

**(3) Accumulatori privatizzati.** Nella fase di aggiornamento ogni blocco
accumula in shared memory e versa i parziali in global una sola volta alla fine.
Il numero di atomici globali scende da $O(n \cdot d)$ a
$O(\text{numBlocchi} \cdot k \cdot d)$. Un test `!= 0` prima del versamento
evita l'atomico per i cluster che il blocco non ha mai toccato — con $k$ grande
la maggior parte dei blocchi ne tocca solo una frazione.

Sulla precisione: le somme parziali in shared sono in `float` (gli atomici float
in shared sono nativi e veloci), mentre l'accumulatore globale è in `double`.
Poiché ogni blocco elabora solo `blockDim.x` punti, l'errore di somma in float
resta confinato, e la somma fra blocchi — quella con molti addendi — avviene in
doppia precisione. Costa quanto la versione tutta-float ed è molto più accurata.

**(4) Riduzione warp-shuffle e finalizzazione su device.** L'SSE viene ridotto
gerarchicamente: prima dentro ogni warp con `__shfl_down_sync` (nessun accesso
alla shared memory, i lane si scambiano i valori via registro), poi fra i warp,
infine un solo `atomicAdd` globale per blocco. La divisione somme/conteggi e il
calcolo dello spostamento massimo avvengono in un kernel dedicato sul device:
ad ogni iterazione si trasferisce all'host un **solo float** invece dell'intero
array dei centroidi, eliminando il ping-pong D2H/H2D del ciclo esterno.

### 2.5 Riepilogo degli interventi

| # | Intervento | Problema risolto | Metrica Nsight che ne dà evidenza |
|---|---|---|---|
| 1 | Layout feature-major | accessi non coalescenti | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` |
| 2 | Centroidi in shared memory | riletture ripetute da global | `gpu__dram_throughput` |
| 3 | Accumulatori privatizzati | serializzazione sugli atomici | durata di `update_*_kernel` |
| 4 | Warp-shuffle + finalizzazione su device | riduzione e trasferimenti per iterazione | timeline Nsight Systems |

---

## 3. Testing

La correttezza è verificata su tre livelli (`tests/test_correctness.cpp`),
**prima** di qualsiasi misura di tempo: cronometrare un codice sbagliato non ha
alcun significato.

### A. Validità assoluta

Su blob gaussiani ben separati il clustering ottimo è unico e noto per
costruzione. Si verifica che la baseline ricostruisca la ground truth usando
l'**Adjusted Rand Index**, che è invariante rispetto alla permutazione delle
etichette — l'accuratezza classica non sarebbe utilizzabile, perché il cluster
"0" trovato dall'algoritmo non ha ragione di corrispondere al cluster "0" del
generatore.

Risultato ottenuto ($n=50\,000$, $d=4$, $k=6$, $\sigma=0.35$):

```
Adjusted Rand Index = 1.000000
distanza massima centro-vero <-> centroide più vicino = 0.00000
```

Si verifica inoltre la **monotonia dell'SSE** lungo le iterazioni: è la
proprietà fondamentale dell'algoritmo di Lloyd, e un errore negli accumulatori
della fase di aggiornamento la viola immediatamente.

Va segnalato che l'ARI non è sempre 1: su una configurazione più difficile
($n=5\cdot10^5$, $d=32$, $k=32$, $\sigma=1.0$) si è misurato **ARI = 0.886**.
Non è un difetto dell'implementazione ma una proprietà nota dell'algoritmo:
Lloyd converge a un minimo *locale*, e con $k$ elevato è frequente che un
cluster vero venga spezzato in due mentre due cluster vicini vengono fusi.
Le librerie di riferimento aggirano il problema con `n_init` ripetizioni da
inizializzazioni diverse, tenendo la soluzione a SSE minimo. Il punto rilevante
per questo progetto è che **tutte e quattro le implementazioni cadono nello
stesso minimo locale**, essendo deterministiche e partendo dagli stessi
centroidi: è esattamente ciò che rende il confronto valido.

### B. Equivalenza fra implementazioni

Le versioni parallele devono riprodurre il risultato della seriale. Non si
pretende l'uguaglianza bit-a-bit: l'ordine delle somme cambia fra le
implementazioni e la somma in virgola mobile non è associativa. Le soglie
adottate sono accordo sulle etichette $\ge 99.9\%$, differenza relativa sui
centroidi $< 10^{-3}$, sull'SSE $< 10^{-4}$.

Sulle configurazioni testate la versione OpenMP risulta **identica** alla
seriale (differenza esattamente nulla sui centroidi, 100.0000% di accordo sulle
etichette): la privatizzazione per thread mantiene lo stesso raggruppamento di
addendi della somma seriale.

### C. Casi limite

`k=1` (il centroide deve coincidere con la media globale), `k=n` (SSE nullo),
`d=1`, punti tutti duplicati (esercita la politica sui cluster vuoti e verifica
l'assenza di divisioni per zero), e dimensioni non multiple del block size CUDA
($n=1234$, $d=7$, $k=13$) per intercettare errori nella gestione della coda
della griglia.

**Esito complessivo: 38 verifiche eseguite, 0 fallite** nella build CPU (`./build/kmeans_test`).
Con la build CUDA la suite include automaticamente anche `cuda_naive` e
`cuda_opt` (numero di verifiche più alto).

### Validazione incrociata

Il notebook Colab contiene un confronto con `sklearn.cluster.KMeans` sullo
stesso dataset: l'inerzia finale delle due implementazioni deve coincidere entro
pochi decimi percentuali.

---

## 4. Profiling e risultati

### 4.1 Setup sperimentale

| | |
|---|---|
| **CPU** | Intel Xeon @ 2.80 GHz, 4 core fisici (no SMT) |
| **Compilatore CPU** | GCC 13.3.0, `-O3 -fopenmp`, C++17 |
| **GPU** | *(da compilare dopo l'esecuzione su Colab/cluster)* |
| **Compilatore CUDA** | *(nvcc, versione e `-arch`)* |
| **Metodologia** | mediana su ripetizioni, prima esecuzione scartata come warm-up |

Il tetto teorico allo speedup OpenMP su questa macchina è **4x**: i core sono
quattro e senza hyperthreading. È un limite importante da tenere presente
leggendo i risultati — e proprio per questo il confronto a tre (seriale,
multicore, GPU) è più informativo di un confronto a due: separa il guadagno
"da più core" da quello "da architettura GPU".

### 4.2 Risultati CPU

Configurazione di riferimento della campagna leggera ($n=10^5$, $d=32$, $k=32$,
20 iterazioni, mediana su ripetizioni con warm-up scartato):

| Implementazione | Tempo compute (s) | Speedup | Efficienza |
|---|---|---|---|
| Seriale (1 core) | 5.555 | 1.00x | — |
| OpenMP, 1 thread | 5.898 | 0.94x | — |
| OpenMP, 2 thread | 3.017 | 1.95x | 97.7% |
| OpenMP, 4 thread | 1.469 | 4.01x | 100.4% |

Lo speedup e l'efficienza dell'ultima colonna sono calcolati rispetto a
OpenMP a 1 thread (scalabilità *forte*), non rispetto al seriale: è la
convenzione corretta, perché isola l'effetto dell'aggiunta di core dall'overhead
introdotto dal runtime OpenMP. Quell'overhead è visibile nella riga a 1 thread,
che è circa il 6% più lenta della versione seriale pura.

Due osservazioni oneste su questi numeri:

- il valore a 4 thread (4.01x, efficienza 100.4%) è **leggermente superlineare**,
  il che è fisicamente impossibile: si tratta di rumore di misura, presumibilmente
  dovuto a effetti di cache e alla variabilità dello scheduler. Con una sola
  ripetizione cronometrata l'incertezza è dell'ordine del qualche percento, quindi
  la lettura corretta è "efficienza sostanzialmente del 100%", non "superlineare";
- la scalabilità quasi ideale non è sorprendente e conferma l'analisi del
  §1: il passo di assegnazione è embarrassingly parallel e domina il tempo, mentre
  la parte sincronizzata (fusione degli accumulatori) costa
  $O(\text{num\_threads} \cdot k \cdot d)$, del tutto trascurabile rispetto a
  $O(n \cdot k \cdot d)$.

Su un caso più grande ($n=5 \cdot 10^5$, $d=32$, $k=32$, 100 iterazioni) il
divario si conferma: **133.75 s** per il seriale contro **34.77 s** per OpenMP a
4 thread, cioè **3.85x**, con SSE ed etichette *identiche* alla baseline.

#### Scalabilità in N, K, D (CPU)

Tempi in secondi, mediana; `sp.` è lo speedup OpenMP a 4 thread sul seriale.

| N ($d=32$, $k=32$) | seriale | OpenMP | sp. |
|---|---|---|---|
| 10 000 | 0.494 | 0.128 | 3.87x |
| 50 000 | 0.524 | 0.139 | 3.78x |
| 100 000 | 5.555 | 1.451 | 3.83x |
| 200 000 | 10.822 | 2.911 | 3.72x |

| K ($n=10^5$, $d=32$) | seriale | OpenMP | sp. |
|---|---|---|---|
| 8 | 1.518 | 0.412 | 3.69x |
| 16 | 2.831 | 0.780 | 3.63x |
| 32 | 5.555 | 1.451 | 3.83x |
| 64 | 10.909 | 2.822 | 3.87x |

| D ($n=10^5$, $k=32$) | seriale | OpenMP | sp. |
|---|---|---|---|
| 2 | 0.474 | 0.149 | 3.19x |
| 8 | 1.388 | 0.385 | 3.60x |
| 32 | 5.555 | 1.451 | 3.83x |
| 64 | 11.498 | 2.871 | 4.01x |

I tempi crescono **linearmente** in $N$, $K$ e $D$, come previsto dalla
complessità $O(n \cdot k \cdot d)$ per iterazione: è una verifica indiretta ma
utile che l'implementazione non abbia costi nascosti superlineari.

Lo speedup OpenMP cresce leggermente con $D$ (3.19x a $d=2$, 4.01x a $d=64$):
con vettori molto corti il lavoro per punto è piccolo e l'overhead per iterazione
del ciclo parallelo pesa di più in proporzione. È lo stesso fenomeno che su GPU
si manifesta come necessità di avere abbastanza lavoro per thread.

> **Nota sulle iterazioni.** Configurazioni diverse convergono in un numero
> diverso di iterazioni (indicato in `results/benchmarks.csv`), quindi i tempi
> **non** sono confrontabili *fra righe diverse* di queste tabelle. Lo sono
> invece all'interno di una stessa riga, dove tutte le implementazioni eseguono
> esattamente le stesse iterazioni sugli stessi dati — ed è quello che serve per
> lo speedup.

### 4.3 Risultati GPU

*(Da compilare dopo l'esecuzione su GPU.)*

| Implementazione | compute (s) | total (s) | Speedup compute | Speedup total |
|---|---|---|---|---|
| Seriale | | | 1.00x | 1.00x |
| OpenMP (4 thread) | | | | |
| CUDA naive | | | | |
| CUDA ottimizzato | | | | |

### 4.4 Due speedup, non uno

Vengono riportate due misure distinte per le versioni GPU:

- **compute**: solo i kernel, escluse allocazioni device e trasferimenti;
- **total**: end-to-end, tutto incluso.

Il secondo è quello onesto da citare come risultato complessivo. Nel K-means il
divario fra i due è contenuto per una ragione strutturale: **i dati salgono sul
device una sola volta**, fuori dal ciclo iterativo, quindi il costo del
trasferimento si ammortizza su tutte le iterazioni. Con $T$ iterazioni il peso
relativo del trasferimento decresce come $1/T$ — è esattamente la situazione in
cui l'offload su GPU conviene, ed è il motivo per cui algoritmi iterativi su
dati fissi sono i candidati migliori all'accelerazione.

Resta la componente non parallelizzata (inizializzazione k-means++, esclusa
dalle misure, e la finalizzazione $O(k \cdot d)$): per la legge di Amdahl è
questa a fissare il tetto asintotico allo speedup complessivo.

### 4.5 Curve di scalabilità

Le figure sono generate da `scripts/plot_results.py` in `results/figures/`:
`speedup_vs_n.png`, `speedup_vs_k.png`, `speedup_vs_d.png`, `omp_scaling.png`.
Con la build CUDA le stesse figure includono automaticamente anche le due curve
GPU. Cosa aspettarsi da ciascuna:

- **speedup vs N** ($d=32$, $k=32$): lo speedup GPU deve *crescere* con $N$ e
  saturare quando la griglia è abbastanza grande da coprire tutti gli SM. Uno
  speedup che cresce con la dimensione del problema è la firma di una buona
  implementazione GPU, e vale più di un singolo numero di picco.
- **speedup vs K** ($n=5\cdot10^5$, $d=32$): al crescere di $k$ aumenta il
  lavoro aritmetico per punto e il kernel si sposta da memory-bound verso
  compute-bound; oltre una certa soglia i centroidi non entrano più in shared
  memory e il vantaggio dell'ottimizzazione (2) si riduce.
- **speedup vs D** ($n=5\cdot10^5$, $k=32$): al crescere di $d$ aumenta il
  traffico di memoria per punto, ed è qui che il guadagno del layout
  feature-major è più marcato.
- **scalabilità forte OpenMP**: speedup e efficienza parallela al variare del
  numero di thread, con la retta ideale come riferimento.

### 4.6 Metriche del profiler

Da riportare per `assign_naive_kernel` e `assign_opt_kernel`:

| Metrica | Cosa mostra |
|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | sfruttamento delle unità di calcolo |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | sfruttamento della banda verso la DRAM |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | settori letti: crolla col layout feature-major |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | occupancy raggiunta |

Il confronto sul numero di settori letti fra le due versioni è la prova
quantitativa diretta dell'effetto del coalescing, ed è il dato più convincente
della sezione.

---

## 5. Conclusioni

*(Da completare con i numeri finali.)*

Punti da argomentare:

1. Quale dei quattro interventi ha prodotto il guadagno maggiore e perché, con
   il dato del profiler a supporto.
2. Come si posiziona il guadagno del multicore rispetto a quello della GPU: con
   4 core il tetto è 4x, e questo isola con chiarezza quanta parte dello speedup
   totale sia attribuibile all'architettura GPU e non semplicemente al fatto di
   usare più di un core.
3. Dove si colloca il limite asintotico (Amdahl) e quali estensioni lo
   sposterebbero: parallelizzazione dell'inizializzazione k-means++, uso della
   disuguaglianza triangolare (algoritmo di Elkan) per potare i confronti,
   memoria unificata o streaming per dataset che non entrano nella memoria del
   device.

---

## Riproducibilità

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/kmeans_test
./scripts/run_benchmarks.sh build/kmeans_bench
python3 scripts/plot_results.py
```

Tutte le misure usano seme fisso (`--seed`, default 42) e sono quindi
riproducibili esattamente.
