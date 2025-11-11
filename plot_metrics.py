import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

LOG_DIR = "/mnt/e/logs-10-11-25/logs/csv"  # folder where all CSVs are stored
OUTPUT_FILE = "fold_metrics.png"

csv_files = sorted(glob.glob(os.path.join(LOG_DIR, "baseline_training_log_model_0_fold_*.csv")))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {LOG_DIR}")

fold_data = {}
for file in csv_files:
    fold_number = int(os.path.basename(file).split("fold_")[1].split("_")[0])
    df = pd.read_csv(file)
    fold_data[fold_number] = df

sample_df = next(iter(fold_data.values()))
metric_cols = [col for col in sample_df.columns[1:] if pd.api.types.is_numeric_dtype(sample_df[col])]

num_metrics = len(metric_cols)
fig, axes = plt.subplots(nrows=(num_metrics + 2) // 3, ncols=3, figsize=(18, 5 * ((num_metrics + 2)//3)))
axes = axes.flatten()

for idx, metric in enumerate(metric_cols):
    ax = axes[idx]
    for fold, df in fold_data.items():
        ax.plot(df.index, df[metric], label=f"Fold {fold}")
    ax.set_title(metric)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.show()

print(f"Plot saved to {OUTPUT_FILE}")
