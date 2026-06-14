"""Benchmark: train every model and produce a comparison report.

Run:  python -m app.benchmark [epochs]

Trains the popularity baseline, MF, GMF, MLP, and NeuMF (plus NeuMF under the
BPR pairwise objective) on the same data and evaluates them with the identical
leave-one-out protocol, then writes:

  * reports/benchmark.md   — a HR@K / NDCG@K comparison table
  * reports/benchmark.png  — NDCG@K training curves + a final-metric bar chart
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import settings
from .data import load_and_prepare
from .training import set_seed, train_model

# Repo-root /reports, regardless of the working directory the command is run from.
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"

# (label, model_name, objective)
CONFIGS = [
    ("Popularity", "popularity", "bce"),
    ("MF", "mf", "bce"),
    ("GMF", "gmf", "bce"),
    ("MLP", "mlp", "bce"),
    ("NeuMF", "neumf", "bce"),
    ("NeuMF (BPR)", "neumf", "bpr"),
]


def run_benchmark(epochs: int) -> dict:
    set_seed(settings.seed)
    print(f"Loading dataset: {settings.dataset}")
    data = load_and_prepare(settings)
    print(f"Users: {data.num_users}  Items: {data.num_items}  Test users: {len(data.test_items)}\n")

    k = settings.top_k
    results, histories = [], {}
    for label, name, objective in CONFIGS:
        set_seed(settings.seed)  # same init/sampling seed for a fair comparison
        print(f"Training {label} ...")
        _model, history, metrics = train_model(name, data, settings, objective, epochs=epochs)
        results.append({"label": label, "objective": objective, **metrics})
        histories[label] = history
        print(f"  -> HR@{k}={metrics[f'hr@{k}']:.4f}  NDCG@{k}={metrics[f'ndcg@{k}']:.4f}\n")

    # NeuMF with popularity-aware (hard) negatives, vs. the uniform-negative NeuMF above.
    set_seed(settings.seed)
    print("Training NeuMF (pop-neg) ...")
    cfg_pop = settings.model_copy(update={"negative_sampling": "popularity"})
    _m, history, metrics = train_model("neumf", data, cfg_pop, "bce", epochs=epochs)
    results.append({"label": "NeuMF (pop-neg)", "objective": "bce", **metrics})
    histories["NeuMF (pop-neg)"] = history
    print(f"  -> HR@{k}={metrics[f'hr@{k}']:.4f}  NDCG@{k}={metrics[f'ndcg@{k}']:.4f}\n")

    # SASRec converges much slower; give it a larger budget (noted in the report).
    sas_epochs = max(epochs * 4, 40)
    set_seed(settings.seed)
    print(f"Training SASRec ({sas_epochs} epochs) ...")
    from .sasrec import train_sasrec
    _m, history, metrics = train_sasrec(data, settings, epochs=sas_epochs)
    results.append({"label": "SASRec", "objective": "bce-seq", **metrics})
    histories["SASRec"] = history
    print(f"  -> HR@{k}={metrics[f'hr@{k}']:.4f}  NDCG@{k}={metrics[f'ndcg@{k}']:.4f}\n")

    return {
        "results": results, "histories": histories, "data": data,
        "epochs": epochs, "sas_epochs": sas_epochs,
    }


def write_markdown(results: list[dict], epochs: int, sas_epochs: int) -> Path:
    k = settings.top_k
    ranked = sorted(results, key=lambda r: r[f"ndcg@{k}"], reverse=True)
    best = ranked[0]["label"]

    lines = [
        f"# Benchmark — {settings.dataset}",
        "",
        f"Leave-one-out evaluation (held-out item vs. {settings.num_eval_negatives} "
        f"sampled negatives). Trained for {epochs} epochs each; same seed, "
        f"embedding dim {settings.gmf_dim}, MLP layers {list(settings.mlp_layers)}.",
        "",
        f"| Model | Objective | HR@{k} | NDCG@{k} |",
        "|---|---|---:|---:|",
    ]
    for r in ranked:
        marker = " **★**" if r["label"] == best else ""
        lines.append(
            f"| {r['label']}{marker} | {r['objective']} | "
            f"{r[f'hr@{k}']:.4f} | {r[f'ndcg@{k}']:.4f} |"
        )
    lines += [
        "",
        f"★ best NDCG@{k}. Higher is better for both metrics.",
        "",
        f"Notes: **NeuMF (pop-neg)** uses popularity-aware (hard) negatives instead "
        f"of uniform. **SASRec** is the sequential model; it converges much more "
        f"slowly, so it was trained for {sas_epochs} epochs (vs {epochs}) — at the "
        f"shared {epochs}-epoch budget it sits near the popularity floor, but given "
        f"more steps it reaches the learned-model range.",
        "",
        "![benchmark](benchmark.png)",
        "",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "benchmark.md"
    path.write_text("\n".join(lines))
    return path


def write_plot(results: list[dict], histories: dict, epochs: int) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot. (pip install -r requirements-dev.txt)")
        return None

    k = settings.top_k
    ndcg_key = f"ndcg@{k}"
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))

    # Popularity has no epoch history, so it's drawn as a flat reference line.
    for label, history in histories.items():
        if len(history) > 1:
            xs = [h["epoch"] for h in history]
            ys = [h[ndcg_key] for h in history]
            ax0.plot(xs, ys, marker="o", markersize=3, label=label)
        else:
            ax0.axhline(history[0][ndcg_key], ls="--", alpha=0.6, label=label)
    ax0.set_title(f"NDCG@{k} over training")
    ax0.set_xlabel("epoch")
    ax0.set_ylabel(f"NDCG@{k}")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.2)

    labels = [r["label"] for r in results]
    xs = range(len(labels))
    width = 0.38
    ax1.bar([x - width / 2 for x in xs], [r[f"hr@{k}"] for r in results], width, label=f"HR@{k}")
    ax1.bar([x + width / 2 for x in xs], [r[ndcg_key] for r in results], width, label=f"NDCG@{k}")
    ax1.set_title("Final ranking metrics")
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "benchmark.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else max(10, settings.epochs // 2)
    out = run_benchmark(epochs)
    md = write_markdown(out["results"], out["epochs"], out["sas_epochs"])
    png = write_plot(out["results"], out["histories"], out["epochs"])
    print(f"Wrote {md}" + (f" and {png}" if png else ""))


if __name__ == "__main__":
    main()
