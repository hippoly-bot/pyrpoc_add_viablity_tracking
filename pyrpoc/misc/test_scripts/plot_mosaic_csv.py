import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_mosaic_data(csv_path):
    df = pd.read_csv(csv_path, index_col=0)

    focus_metrics = None
    if "Focus Metric" in df.columns:
        focus_metrics = df["Focus Metric"]
        df = df.drop(columns=["Focus Metric"])

    plt.figure(figsize=(8, 5))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col, alpha=0.7)
    plt.xlabel("Repetition Index")
    plt.ylabel("Average Intensity")
    plt.title("Photobleaching Decay Curves")
    plt.legend(loc='best', fontsize="small", frameon=False)
    plt.tight_layout()
    plt.show()

    if focus_metrics is not None:
        plt.figure(figsize=(8, 3))
        plt.plot(focus_metrics.values, marker='o')
        plt.xlabel("Tile Index (Order of Acquisition)")
        plt.ylabel("Focus Metric")
        plt.title("Autofocus Quality Metrics")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plot_mosaic_data(r'mosaic_data.csv')
