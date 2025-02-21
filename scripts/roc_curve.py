import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_roc_curves(true_labels, pred_probabilities, class_names, save_path):
    """Plot ROC curves for multi-class classification."""
    n_classes = len(class_names)
    true_binary = label_binarize(true_labels, classes=range(n_classes))

    plt.figure(figsize=(10, 8))

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#FFC107']
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(true_binary[:, i], pred_probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], linewidth=2,
                 label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Eye Disease Classification')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curves saved to {save_path}")
