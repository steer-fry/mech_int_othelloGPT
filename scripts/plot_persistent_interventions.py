"""Plot saved persistence results; requires matplotlib, independent of inference."""
import argparse
import json
import math
from pathlib import Path


def plot_ordinary_predictions(directory, summary, plt):
    """Plot the distinct prediction metrics measured on ordinary legal games."""
    if not summary or "top1_legal_percent" not in summary[0]:
        return
    fig, ax = plt.subplots(figsize=(9, 5.6))
    horizons = [row["horizon"] for row in summary]
    for metric, label, color, linestyle in (
        ("top1_legal_percent", "Top-1 prediction is legal", "#15803d", "--"),
        ("exact_set_percent", "Exact legal-move set", "#2563eb", "-"),
        ("next_token_percent", "Matches sampled next token", "#9333ea", "-"),
        ("mean_uniform_next_token_chance", "Uniform-sampling match chance", "#6b7280", "--"),
    ):
        values = [row[metric] for row in summary]
        if metric == "mean_uniform_next_token_chance":
            values = [100 * value if value is not None else None for value in values]
        ax.plot(horizons, values, label=label, color=color, linestyle=linestyle,
                marker="s" if metric == "top1_legal_percent" else "o", markersize=4,
                zorder=3 if metric == "top1_legal_percent" else 2)
    ax.set(
        title="Unsteered model on ordinary legal games", ylabel="Accuracy or match chance (%)",
        xlabel="Additional moves after original prefix", ylim=(-2, 102),
        xticks=range(0, max(horizons) + 1, 2 if max(horizons) > 6 else 1),
    )
    ax.grid(alpha=.2)
    ax.legend(frameon=False, loc="center right", fontsize=9)
    fig.text(
        .5, .035,
        "Targets are uniformly sampled legal moves; matching a target by chance averages 1 / legal-move count.\n"
        "Top-1 legality accepts any legal move. Exact legal-set accuracy requires recovering all legal moves.",
        ha="center", va="bottom", fontsize=9, color="#374151",
    )
    fig.tight_layout(rect=(0, .13, 1, 1))
    fig.savefig(directory / "next_token_accuracy.png", dpi=180)
    fig.savefig(directory / "next_token_accuracy.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edited_results = json.loads((args.directory / "results.json").read_text())
    summary = edited_results["summary"]
    control_path = args.directory / "ordinary_game_control" / "results.json"
    control_summary = []
    if control_path.exists():
        control_results = json.loads(control_path.read_text())
        edited_config, control_config = edited_results["config"], control_results["config"]
        for key in (
            "checkpoint_sha256", "benchmark_sha256", "evaluated_records", "limited_run", "precision",
            "seed", "horizon", "max_input_length", "model_implementation_sha256",
        ):
            if key not in edited_config or key not in control_config:
                raise ValueError(f"Cannot compare control: config field {key!r} is missing")
            if edited_config[key] != control_config[key]:
                raise ValueError(f"Cannot compare control: config field {key!r} differs")
        control_summary = control_results["summary"]
        control_summary = sorted(
            (row for row in control_summary
             if row["action"] == "ordinary_game" and row["method"] == "clean"),
            key=lambda row: row["horizon"],
        )
        if not control_summary:
            raise ValueError("Cannot compare control: ordinary_game/clean summary rows are missing")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True, sharey="row")
    max_horizon = max(row["horizon"] for row in summary + control_summary)
    max_errors = max(row["mean_errors"] for row in summary + control_summary
                     if row["mean_errors"] is not None)
    error_limit = max(1, math.ceil(max_errors))
    for column, action in enumerate(("flip", "erase")):
        series = []
        for method, label, color in (
            ("patched", "Patched → edited board", "#2563eb"),
            ("baseline", "Unsteered → edited board", "#6b7280"),
        ):
            rows = sorted(
                (row for row in summary if row["action"] == action and row["method"] == method),
                key=lambda row: row["horizon"],
            )
            series.append((rows, label, color, "-"))
        if control_summary:
            series.append((control_summary, "Unsteered → original board", "#15803d", "--"))
        for rows, label, color, linestyle in series:
            for metric_row, metric in enumerate(("exact_set_percent", "mean_errors")):
                axes[metric_row, column].plot(
                    [row["horizon"] for row in rows], [row[metric] for row in rows],
                    marker="o", markersize=4, label=label, color=color, linestyle=linestyle,
                )
        axes[0, column].set(title=action.upper(), ylim=(-2, 102))
        axes[1, column].set(
            xlabel="Additional moves after original prefix", ylim=(-.02 * error_limit, error_limit),
            xticks=range(0, max_horizon + 1, 2 if max_horizon > 6 else 1),
        )
        for ax in axes[:, column]:
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("Exact legal-set accuracy (%)")
    axes[1, 0].set_ylabel("Mean legal-set errors (FP + FN)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94),
               ncol=len(labels), frameon=False)
    fig.suptitle("Persistence after one-time steering", fontsize=13, y=.985)
    if control_summary:
        note = (
            "Patched and unsteered edited-board runs share continuations. The original-board control\n"
            "uses independent legal continuations, so its gap is not a paired intervention effect."
        )
    else:
        note = "Patched and unsteered runs share continuations from the edited board."
    fig.text(.5, .045, note, ha="center", va="bottom", fontsize=9, color="#374151")
    fig.subplots_adjust(left=.075, right=.99, top=.84, bottom=.17, hspace=.16, wspace=.04)
    fig.savefig(args.directory / "persistence.png", dpi=180)
    fig.savefig(args.directory / "persistence.svg")
    plt.close(fig)
    plot_ordinary_predictions(control_path.parent, control_summary, plt)


if __name__ == "__main__":
    main()
