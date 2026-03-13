import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def add_labels(ax, fmt="%.4f"):
    """Helper to add labels to the top of bars and adjust Y-axis height."""
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=3, fontsize=10)

    # Adjust Y-axis top limit so labels fit (add 10% headroom)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.1)


def save_experiment_plots(df, output_root):
    # 1. Setup Styling
    sns.set_theme(style="whitegrid")

    # 2. Pre-processing
    df.columns = df.columns.str.strip().str.upper()
    df["METRIC_CLEAN"] = df["METRIC"].str.lower().str.strip()

    print(f"Generating plots in: {output_root}")

    # Define metrics where "higher is better" (Accuracy, F1, Exact Match, etc.)
    higher_is_better_metrics = ["acc", "acc_norm", "accuracy", "exact_match", "f1"]

    # --- 1. INDIVIDUAL TASK PLOTS ---
    acc_df = df[df["METRIC_CLEAN"].isin(higher_is_better_metrics)].copy()

    if not acc_df.empty:
        # FILTER: If a task has 'acc_norm', drop the 'acc' metric for that task
        tasks_with_acc_norm = acc_df[acc_df["METRIC_CLEAN"] == "acc_norm"]["TASK"].unique()
        acc_df = acc_df[~((acc_df["TASK"].isin(tasks_with_acc_norm)) & (acc_df["METRIC_CLEAN"] == "acc"))]

        # Group by both TASK and METRIC
        for task in acc_df["TASK"].unique():
            for metric in acc_df[acc_df["TASK"] == task]["METRIC_CLEAN"].unique():
                task_data = acc_df[(acc_df["TASK"] == task) & (acc_df["METRIC_CLEAN"] == metric)]

                # Create Folder for Task
                safe_task = f"{task}_{metric}".replace("/", "_").replace(" ", "_")
                task_dir = output_root / "Scores_Per_Task" / safe_task
                task_dir.mkdir(parents=True, exist_ok=True)

                for fs in task_data["NUM_FEWSHOT"].unique():
                    is_nan = pd.isna(fs)
                    fs_label = "NaN" if is_nan else str(int(fs))

                    # Filter by Few-shot count
                    plot_df = (
                        task_data[task_data["NUM_FEWSHOT"].isna()]
                        if is_nan
                        else task_data[task_data["NUM_FEWSHOT"] == fs]
                    )
                    if plot_df.empty:
                        continue

                    # Plot
                    plt.figure(figsize=(12, 7))
                    ax = sns.barplot(
                        data=plot_df.sort_values("SCORE", ascending=False),
                        x="EXPERIMENT",
                        y="SCORE",
                        palette="viridis",
                    )

                    add_labels(ax, fmt="%.4f")

                    plt.title(f"{metric.upper()}: {task} (Few-shot: {fs_label})")
                    plt.xticks(rotation=45, ha="right")
                    plt.ylabel("Score")
                    plt.tight_layout()
                    plt.savefig(task_dir / f"fewshot_{fs_label}.png")
                    plt.close()

    # --- 2. AVERAGE SCORE (Aggregated) ---
    if not acc_df.empty:
        avg_acc_dir = output_root / "Average_Score"
        avg_acc_dir.mkdir(parents=True, exist_ok=True)

        # Group by Fewshot + Experiment -> Mean Score
        # (This now safely averages only acc_norm for tasks that had both)
        avg_df = (
            acc_df.groupby(["NUM_FEWSHOT", "EXPERIMENT"], dropna=False)["SCORE"]
            .mean()
            .reset_index()
        )

        for fs in avg_df["NUM_FEWSHOT"].unique():
            is_nan = pd.isna(fs)
            fs_label = "NaN" if is_nan else str(int(fs))

            plot_df = (
                avg_df[avg_df["NUM_FEWSHOT"].isna()]
                if is_nan
                else avg_df[avg_df["NUM_FEWSHOT"] == fs]
            )
            if plot_df.empty:
                continue

            plt.figure(figsize=(12, 7))
            ax = sns.barplot(
                data=plot_df.sort_values("SCORE", ascending=False),
                x="EXPERIMENT",
                y="SCORE",
                palette="mako",
            )

            add_labels(ax, fmt="%.4f")

            plt.title(f"Average Score Across All Tasks (Few-shot: {fs_label})")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Mean Score")
            plt.tight_layout()
            plt.savefig(avg_acc_dir / f"average_fewshot_{fs_label}.png")
            plt.close()

    # --- 3. PERPLEXITY ---
    ppl_df = df[df["METRIC_CLEAN"] == "perplexity"].copy()

    if not ppl_df.empty:
        ppl_dir = output_root / "Perplexity"
        ppl_dir.mkdir(parents=True, exist_ok=True)

        for task in ppl_df["TASK"].unique():
            task_plot_df = ppl_df[ppl_df["TASK"] == task].sort_values(
                "SCORE", ascending=True
            )

            plt.figure(figsize=(12, 7))
            ax = sns.barplot(
                data=task_plot_df, x="EXPERIMENT", y="SCORE", palette="rocket"
            )

            add_labels(ax, fmt="%.3f") 

            plt.title(f"Perplexity: {task}")
            plt.ylabel("Perplexity (Lower is Better)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            safe_task = str(task).replace("/", "_").replace(" ", "_")
            plt.savefig(ppl_dir / f"{safe_task}.png")
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate labeled plots from a Parquet benchmark file."
    )
    parser.add_argument(
        "input", type=str, help="Path to the .parquet file containing the data"
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()

    if not input_path.exists():
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)

    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        sys.exit(1)

    output_dir = input_path.parent / "plots"
    save_experiment_plots(df, output_dir)
    print(f"Done. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()