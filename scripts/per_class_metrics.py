import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


def plot_per_class_metrics(true_labels, pred_labels, class_names, save_path):
    """Plot per-class precision, recall, and F1-score."""
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None
    )

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2196F3')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#4CAF50')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#FF9800')

    ax.set_xlabel('Disease Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Classification Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Per-class metrics plot saved to {save_path}")
