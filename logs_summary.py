import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

folder = "logs/baseline_1st"

csv_files = glob.glob(os.path.join(folder, "baseline_training_log_model_*_fold_*_*.csv"))
csv_files.sort()

metrics = ["auc", "binary_accuracy", "loss", "precision", "recall"]

for file_path in csv_files:
    df = pd.read_csv(file_path)

    fold_name = os.path.basename(file_path).split("_fold_")[1].split("_")[0]

    pre_df = df.iloc[:5]
    pseudo_df = df.iloc[5:10]

    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            axes[i].set_visible(False)
            continue

        ax = axes[i]

        ax.plot(pre_df["epoch"], pre_df[metric], marker="o", color="blue", label="Pretraining")
        ax.plot(pseudo_df["epoch"], pseudo_df[metric], marker="o", color="orange", label="Pseudo-label training")

        ax.set_title(f"{metric} per phase (Fold {fold_name})")
        ax.set_xlabel("Epoch (0–4)")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend()

    plt.suptitle(f"Training Metrics — Fold {fold_name}\n(Blue: Pretraining | Orange: Pseudo-labeling)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

    plt.savefig(f"scores/plots/fold_{fold_name}_metrics.png", dpi=200)
