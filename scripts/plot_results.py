#!/usr/bin/env python3
"""Genera i grafici di speedup e scalabilita' a partire dai CSV di benchmark.

Uso:
    python3 scripts/plot_results.py [--csv results/benchmarks.csv]
                                    [--omp-csv results/omp_scaling.csv]
                                    [--outdir results/figures]

Produce quattro figure, una per ciascuno studio prodotto da run_benchmarks.sh:
    speedup_vs_n.png   speedup rispetto al seriale al crescere di N
    speedup_vs_k.png   idem al crescere del numero di cluster
    speedup_vs_d.png   idem al crescere della dimensionalita'
    omp_scaling.png    scalabilita' forte OpenMP con la retta ideale

Dipendenze: pandas e matplotlib (nessuna delle due e' necessaria per compilare
o eseguire il progetto: servono solo per la reportistica).
"""

import argparse
import os
import sys

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # backend non interattivo: funziona anche via SSH
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Dipendenza mancante ({exc}). Installa con: pip install pandas matplotlib")


# Ordine e stile fissi, cosi' che tutte le figure del report siano coerenti.
STYLE = {
    "serial":     dict(label="Seriale (1 core)",     marker="o", color="#4b5563"),
    "omp":        dict(label="OpenMP (multicore)",   marker="s", color="#2563eb"),
    "cuda_naive": dict(label="CUDA naive",           marker="^", color="#f59e0b"),
    "cuda_opt":   dict(label="CUDA ottimizzato",     marker="D", color="#16a34a"),
}
ORDER = ["serial", "omp", "cuda_naive", "cuda_opt"]


def add_speedup(df):
    """Aggiunge le colonne di speedup, normalizzando rispetto al seriale
    all'interno di ciascuna configurazione (n, d, k)."""
    # La configurazione di riferimento ricorre in piu' sweep (e' il punto in cui
    # le curve si incrociano), quindi compare piu' volte nel CSV. Si consolidano
    # le righe duplicate prendendo la mediana dei tempi: senza questo passaggio
    # il join con la baseline duplicherebbe le righe e i grafici mostrerebbero
    # due punti sovrapposti alla stessa ascissa.
    df = (df.groupby(["impl", "n", "d", "k", "threads"], as_index=False)
            .agg(iterations=("iterations", "first"),
                 inertia=("inertia", "first"),
                 time_compute_s=("time_compute_s", "median"),
                 time_total_s=("time_total_s", "median")))

    base = (df[df["impl"] == "serial"]
            .set_index(["n", "d", "k"])[["time_compute_s", "time_total_s"]]
            .rename(columns={"time_compute_s": "base_compute",
                             "time_total_s": "base_total"}))
    df = df.join(base, on=["n", "d", "k"])
    df["speedup_compute"] = df["base_compute"] / df["time_compute_s"]
    df["speedup_total"] = df["base_total"] / df["time_total_s"]
    return df


def plot_sweep(df, xcol, fixed, title, xlabel, path):
    """Traccia lo speedup in funzione di xcol, tenendo fissi gli altri parametri."""
    sub = df.copy()
    for col, val in fixed.items():
        sub = sub[sub[col] == val]
    if sub.empty:
        print(f"  (saltato {os.path.basename(path)}: nessun dato con {fixed})")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for impl in ORDER:
        s = sub[sub["impl"] == impl].sort_values(xcol)
        if s.empty:
            continue
        ax.plot(s[xcol], s["speedup_compute"], **STYLE[impl])

    ax.axhline(1.0, color="#9ca3af", linestyle=":", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Speedup rispetto al seriale")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  scritto {path}")


def plot_omp_scaling(path_csv, path_png):
    if not os.path.exists(path_csv):
        print(f"  (saltato {os.path.basename(path_png)}: {path_csv} assente)")
        return
    df = pd.read_csv(path_csv)
    df = df[df["impl"] == "omp"].sort_values("threads")
    if df.empty:
        return

    t1 = df[df["threads"] == 1]["time_compute_s"]
    if t1.empty:
        print("  (saltato omp_scaling: manca la misura a 1 thread)")
        return
    base = float(t1.iloc[0])
    df = df.assign(speedup=base / df["time_compute_s"])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(df["threads"], df["speedup"], marker="s", color="#2563eb", label="Misurato")
    ax.plot(df["threads"], df["threads"], linestyle="--", color="#9ca3af", label="Ideale (lineare)")
    ax.set_xlabel("Numero di thread OpenMP")
    ax.set_ylabel("Speedup rispetto a 1 thread")
    ax.set_title("Scalabilita' forte della versione OpenMP")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    print(f"  scritto {path_png}")

    # Efficienza parallela: utile da citare nel report accanto al grafico.
    print("\n  Efficienza parallela OpenMP:")
    for _, row in df.iterrows():
        eff = row["speedup"] / row["threads"] * 100.0
        print(f"    {int(row['threads']):2d} thread: speedup {row['speedup']:5.2f}x"
              f"  efficienza {eff:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/benchmarks.csv")
    ap.add_argument("--omp-csv", default="results/omp_scaling.csv")
    ap.add_argument("--outdir", default="results/figures")
    # Configurazione di riferimento: deve coincidere con REF_N/REF_D/REF_K
    # usati da run_benchmarks.sh, altrimenti gli sweep in K e D restano vuoti.
    ap.add_argument("--ref-n", type=int, default=500000)
    ap.add_argument("--ref-d", type=int, default=32)
    ap.add_argument("--ref-k", type=int, default=32)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"File non trovato: {args.csv}\nEsegui prima ./scripts/run_benchmarks.sh")

    os.makedirs(args.outdir, exist_ok=True)
    df = add_speedup(pd.read_csv(args.csv))

    print("Generazione figure:")
    plot_sweep(df, "n", {"d": args.ref_d, "k": args.ref_k},
               f"Speedup al crescere della dimensione del dataset "
               f"(d={args.ref_d}, k={args.ref_k})",
               "Numero di punti N", os.path.join(args.outdir, "speedup_vs_n.png"))
    plot_sweep(df, "k", {"n": args.ref_n, "d": args.ref_d},
               f"Speedup al crescere del numero di cluster "
               f"(n={args.ref_n:.0e}, d={args.ref_d})",
               "Numero di cluster K", os.path.join(args.outdir, "speedup_vs_k.png"))
    plot_sweep(df, "d", {"n": args.ref_n, "k": args.ref_k},
               f"Speedup al crescere della dimensionalita' "
               f"(n={args.ref_n:.0e}, k={args.ref_k})",
               "Dimensionalita' D", os.path.join(args.outdir, "speedup_vs_d.png"))
    plot_omp_scaling(args.omp_csv, os.path.join(args.outdir, "omp_scaling.png"))

    # Tabella riassuntiva pronta da incollare nel report.
    print(f"\nTabella riassuntiva (configurazione di riferimento "
          f"n={args.ref_n}, d={args.ref_d}, k={args.ref_k}):")
    ref = df[(df["n"] == args.ref_n) & (df["d"] == args.ref_d) & (df["k"] == args.ref_k)]
    if ref.empty:
        print("  nessuna riga per la configurazione di riferimento")
        return
    print(f"  {'impl':<12} {'compute (s)':>12} {'total (s)':>12} "
          f"{'sp. compute':>12} {'sp. total':>10}")
    for impl in ORDER:
        row = ref[ref["impl"] == impl]
        if row.empty:
            continue
        r = row.iloc[0]
        print(f"  {impl:<12} {r['time_compute_s']:>12.4f} {r['time_total_s']:>12.4f} "
              f"{r['speedup_compute']:>11.2f}x {r['speedup_total']:>9.2f}x")


if __name__ == "__main__":
    main()
