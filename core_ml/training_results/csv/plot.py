import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict

# Folder containing the CSV files
metrics_dir = "."  # Change this to your actual path

# Regex to parse filenames
pattern = re.compile(r"(.*)_noise_(\d\.\d)_metrics\.csv")

# Group files by noise level
noise_groups = defaultdict(list)

for file in os.listdir(metrics_dir):
    match = pattern.match(file.strip("'"))
    if match:
        loss_fn, noise = match.groups()
        noise_groups[noise].append((loss_fn.strip(), os.path.join(metrics_dir, file)))

# Plotting
for noise, files in sorted(noise_groups.items()):
    plt.figure(figsize=(10, 6))
    for loss_fn, path in sorted(files):
        df = pd.read_csv(path)
        plt.plot(df['epoch'], df['test_acc'], label=loss_fn)

    plt.title(f"Test Accuracy over Epochs (Noise={noise})", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.ylim(0, 1)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"accuracy_plot_noise_{noise}.png")  # Save the plot
    plt.show()  # Or comment this out if running headless
