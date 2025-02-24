import os
import json
from datetime import datetime


class ExperimentLogger:
    """Log training experiments with hyperparameters and results."""

    def __init__(self, log_dir="results/experiments"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log_experiment(self, config, results):
        """Log a single experiment."""
        experiment = {
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'results': results
        }

        log_file = os.path.join(self.log_dir, "experiment_log.jsonl")
        with open(log_file, 'a') as f:
            f.write(json.dumps(experiment) + '\n')

        return experiment

    def get_best_experiment(self, metric='val_accuracy'):
        """Find the best experiment based on a metric."""
        log_file = os.path.join(self.log_dir, "experiment_log.jsonl")
        if not os.path.exists(log_file):
            return None

        best = None
        best_metric = -float('inf')

        with open(log_file, 'r') as f:
            for line in f:
                exp = json.loads(line.strip())
                if exp['results'].get(metric, 0) > best_metric:
                    best_metric = exp['results'][metric]
                    best = exp

        return best

    def summary(self):
        """Print summary of all experiments."""
        log_file = os.path.join(self.log_dir, "experiment_log.jsonl")
        if not os.path.exists(log_file):
            print("No experiments logged yet.")
            return

        experiments = []
        with open(log_file, 'r') as f:
            for line in f:
                experiments.append(json.loads(line.strip()))

        print(f"Total experiments: {len(experiments)}")
        for i, exp in enumerate(experiments):
            print(f"\n--- Experiment {i+1} ---")
            print(f"  Time: {exp['timestamp']}")
            print(f"  Config: {exp['config']}")
            print(f"  Results: {exp['results']}")
