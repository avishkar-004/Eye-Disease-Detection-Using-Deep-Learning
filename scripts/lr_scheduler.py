import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def cosine_decay_schedule(epoch, total_epochs=50, initial_lr=0.001, min_lr=1e-7):
    """Cosine decay learning rate schedule."""
    cosine_decay = 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
    return min_lr + (initial_lr - min_lr) * cosine_decay


def warmup_cosine_schedule(epoch, total_epochs=50, warmup_epochs=5,
                            initial_lr=0.001, min_lr=1e-7):
    """Warmup + cosine decay learning rate schedule."""
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    return cosine_decay_schedule(
        epoch - warmup_epochs, total_epochs - warmup_epochs,
        initial_lr, min_lr
    )


def plot_lr_schedule(schedule_fn, total_epochs=50, save_path=None):
    """Plot learning rate schedule."""
    epochs = range(total_epochs)
    lrs = [schedule_fn(e) for e in epochs]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    plot_lr_schedule(
        lambda e: warmup_cosine_schedule(e),
        save_path="results/lr_schedule.png"
    )
    print("Learning rate schedule plotted.")
