import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera relatório e gráficos a partir das métricas do treino."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.out_dir / "metrics.csv"
    run_path = args.out_dir / "run.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Métricas não encontradas: {metrics_path}")

    with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        rows = list(csv.DictReader(metrics_file))
    if not rows:
        raise ValueError(f"O arquivo de métricas está vazio: {metrics_path}")

    for row in rows:
        for key in row:
            row[key] = float(row[key])

    best = min(rows, key=lambda row: row["val_loss"])
    report = {
        "evaluations": len(rows),
        "best_step": int(best["step"]),
        "best_val_loss": best["val_loss"],
        "best_val_perplexity": math.exp(best["val_loss"]),
        "final_train_loss": rows[-1]["train_loss"],
        "final_val_loss": rows[-1]["val_loss"],
        "final_val_perplexity": math.exp(rows[-1]["val_loss"]),
        "peak_gpu_memory_mb": max(row["gpu_memory_mb"] for row in rows),
        "duration_seconds": rows[-1]["elapsed_seconds"],
    }
    if run_path.exists():
        with run_path.open(encoding="utf-8") as run_file:
            report["run"] = json.load(run_file)

    report_path = args.out_dir / "benchmark.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Instale matplotlib para gerar o gráfico: pip install matplotlib"
        ) from error

    steps = [row["step"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps, [row["train_loss"] for row in rows], label="treino")
    axes[0].plot(steps, [row["val_loss"] for row in rows], label="validação")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Etapa")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, [row["learning_rate"] for row in rows])
    axes[1].set_title("Learning rate")
    axes[1].set_xlabel("Etapa")
    axes[1].set_ylabel("Learning rate")
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(args.out_dir / "training_metrics.png", dpi=160)
    plt.close(figure)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Relatório salvo em: {report_path}")
    print(f"Gráfico salvo em: {args.out_dir / 'training_metrics.png'}")


if __name__ == "__main__":
    main()
