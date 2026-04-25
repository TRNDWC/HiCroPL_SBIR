import argparse
import csv
import json
import importlib
from pathlib import Path


def _import_pyplot():
    try:
        return importlib.import_module("matplotlib.pyplot")
    except Exception as exc:
        raise ImportError("matplotlib is required. Install with: pip install matplotlib") from exc


def _import_event_accumulator():
    try:
        module = importlib.import_module("tensorboard.backend.event_processing.event_accumulator")
        return module.EventAccumulator, module.SCALARS
    except Exception as exc:
        raise ImportError("tensorboard is required. Install with: pip install tensorboard") from exc


def find_event_file(logdir: Path) -> Path:
    candidates = sorted(logdir.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event file found under: {logdir}")
    return candidates[-1]


def load_scalars(event_file: Path):
    EventAccumulator, SCALARS = _import_event_accumulator()
    ea = EventAccumulator(
        str(event_file),
        size_guidance={SCALARS: 0},
    )
    ea.Reload()
    return ea


def smooth_values(values, smooth):
    if smooth <= 0:
        return values
    alpha = max(1e-6, min(1.0, 1.0 - smooth))
    out = []
    running = values[0]
    for v in values:
        running = alpha * v + (1.0 - alpha) * running
        out.append(running)
    return out


def pick_default_metrics(tags):
    available = set(tags)
    preferred = [
        "train_loss_epoch",
        "train_loss",
        "loss",
        "val_mAP",
        "val_map_200",
        "val_map_all",
        "val_P@200",
        "val_P@100",
        "mAP",
    ]
    selected = [m for m in preferred if m in available]
    if selected:
        return selected
    return tags[:8]


def plot_learning_curves(ea, metrics, out_file: Path, smooth=0.0, title="Learning Curve"):
    plt = _import_pyplot()
    tags = ea.Tags().get("scalars", [])
    available = set(tags)

    selected_metrics = list(metrics)
    if not selected_metrics:
        selected_metrics = pick_default_metrics(tags)

    if not selected_metrics:
        raise RuntimeError("No scalar metrics found in event file.")

    missing = [m for m in selected_metrics if m not in available]
    if missing:
        raise ValueError(f"Metrics not found: {missing}. Available metrics: {tags}")

    plt.figure(figsize=(10, 6))
    for metric in selected_metrics:
        events = ea.Scalars(metric)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        if values and smooth > 0:
            values = smooth_values(values, smooth)
        plt.plot(steps, values, label=metric)

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close()


def split_train_val_metrics(tags):
    train_metrics = []
    val_metrics = []
    other_metrics = []
    for tag in tags:
        if tag.startswith("train") or tag == "loss":
            train_metrics.append(tag)
        elif tag.startswith("val") or tag.startswith("mAP") or tag.startswith("P@"):
            val_metrics.append(tag)
        else:
            other_metrics.append(tag)
    return train_metrics, val_metrics, other_metrics


def export_scalars_csv(ea, metrics, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "step", "value"])
        for metric in metrics:
            for e in ea.Scalars(metric):
                writer.writerow([metric, e.step, e.value])


def metric_summary(ea, metrics):
    summary = {}
    for metric in metrics:
        events = ea.Scalars(metric)
        if not events:
            continue
        values = [e.value for e in events]
        steps = [e.step for e in events]
        last_idx = len(values) - 1
        best_max_idx = max(range(len(values)), key=lambda i: values[i])
        best_min_idx = min(range(len(values)), key=lambda i: values[i])
        summary[metric] = {
            "final_step": int(steps[last_idx]),
            "final_value": float(values[last_idx]),
            "best_max_step": int(steps[best_max_idx]),
            "best_max_value": float(values[best_max_idx]),
            "best_min_step": int(steps[best_min_idx]),
            "best_min_value": float(values[best_min_idx]),
            "num_points": len(values),
        }
    return summary


def write_markdown_summary(summary, out_md: Path, event_file: Path):
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Training Report")
    lines.append("")
    lines.append(f"- Event file: {event_file}")
    lines.append("")
    lines.append("## Metric Summary")
    lines.append("")
    lines.append("| Metric | Final | Best Max (step) | Best Min (step) | Points |")
    lines.append("|---|---:|---:|---:|---:|")
    for metric in sorted(summary.keys()):
        s = summary[metric]
        lines.append(
            f"| {metric} | {s['final_value']:.6f} (#{s['final_step']}) | "
            f"{s['best_max_value']:.6f} (#{s['best_max_step']}) | "
            f"{s['best_min_value']:.6f} (#{s['best_min_step']}) | "
            f"{s['num_points']} |"
        )
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def write_json_summary(summary, out_json: Path, event_file: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event_file": str(event_file),
        "metrics": summary,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_report(ea, tags, event_file: Path, report_dir: Path, smooth=0.0):
    report_dir.mkdir(parents=True, exist_ok=True)
    train_metrics, val_metrics, _ = split_train_val_metrics(tags)

    # Keep plots readable by selecting key subsets.
    train_plot_metrics = [m for m in ["train_loss_epoch", "train_loss", "loss"] if m in train_metrics]
    if not train_plot_metrics:
        train_plot_metrics = train_metrics[:6]

    val_plot_metrics = [
        m
        for m in ["val_mAP", "val_map_200", "val_map_all", "val_P@200", "val_P@100", "mAP"]
        if m in val_metrics
    ]
    if not val_plot_metrics:
        val_plot_metrics = val_metrics[:6]

    if train_plot_metrics:
        plot_learning_curves(
            ea,
            train_plot_metrics,
            report_dir / "train_curve.png",
            smooth=smooth,
            title="Training Curves",
        )

    if val_plot_metrics:
        plot_learning_curves(
            ea,
            val_plot_metrics,
            report_dir / "validation_curve.png",
            smooth=smooth,
            title="Validation Curves",
        )

    export_scalars_csv(ea, tags, report_dir / "all_scalars.csv")
    summary = metric_summary(ea, tags)
    write_markdown_summary(summary, report_dir / "summary.md", event_file)
    write_json_summary(summary, report_dir / "summary.json", event_file)

    print(f"Saved report folder: {report_dir}")
    if train_plot_metrics:
        print(f"- Train plot: {report_dir / 'train_curve.png'}")
    if val_plot_metrics:
        print(f"- Validation plot: {report_dir / 'validation_curve.png'}")
    print(f"- CSV scalars: {report_dir / 'all_scalars.csv'}")
    print(f"- Markdown summary: {report_dir / 'summary.md'}")
    print(f"- JSON summary: {report_dir / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="Plot learning curves from TensorBoard event files")
    parser.add_argument("--event_file", type=str, default="", help="Path to a TensorBoard event file")
    parser.add_argument("--logdir", type=str, default="", help="TensorBoard log directory (auto-pick latest event file)")
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=None,
        help="Metric tags to plot. If omitted, auto-select common tags.",
    )
    parser.add_argument("--out", type=str, default="outputs/learning_curve.png", help="Output image path")
    parser.add_argument("--report_dir", type=str, default="", help="Output folder for report-ready artifacts")
    parser.add_argument("--smooth", type=float, default=0.0, help="EMA smoothing strength in [0, 0.99]")
    parser.add_argument("--list", action="store_true", help="Only list available scalar tags")
    args = parser.parse_args()

    if not args.event_file and not args.logdir:
        raise ValueError("Provide either --event_file or --logdir")

    if args.event_file:
        event_file = Path(args.event_file)
    else:
        event_file = find_event_file(Path(args.logdir))

    if not event_file.exists():
        raise FileNotFoundError(f"Event file not found: {event_file}")

    ea = load_scalars(event_file)
    tags = ea.Tags().get("scalars", [])

    print(f"Using event file: {event_file}")
    print("Available scalar tags:")
    for t in tags:
        print(f"- {t}")

    if args.list:
        return

    smooth = max(0.0, min(0.99, args.smooth))

    if args.report_dir:
        build_report(
            ea=ea,
            tags=tags,
            event_file=event_file,
            report_dir=Path(args.report_dir),
            smooth=smooth,
        )
    else:
        metrics = args.metrics if args.metrics is not None else []
        out_file = Path(args.out)
        plot_learning_curves(ea, metrics, out_file, smooth=smooth)
        print(f"Saved learning curve to: {out_file}")


if __name__ == "__main__":
    main()