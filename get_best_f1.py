import argparse
import pickle
import os

def load_best_f1(path, metric_key="f1_score", monitor="accuracy"):
    """
    path: path to an out_finetune pickle file
    metric_key: which metric to grab from the metric dicts ('f1_score' in your case)
    monitor: which validation metric to use to pick the best epoch ('accuracy' in your code)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, "rb") as f:
        _, _, metric_list = pickle.load(f)

    best_idx = None
    best_test_idx = None
    best_val = float("-inf")
    best_test = float("-inf")

    for i, (train_m, valid_m, test_m) in enumerate(metric_list):
        val_metric = valid_m.get(monitor, None)
        if val_metric is None:
            raise KeyError(f"Validation metric '{monitor}' not found in metrics: {valid_m.keys()}")
        if val_metric > best_val:
            best_val = val_metric
            best_test  = val_metric
            best_idx = i

    best_train_m, best_valid_m, best_test_m = metric_list[best_idx]

    f1_test = best_test_m.get(metric_key, None)
    if f1_test is None:
        raise KeyError(f"Metric '{metric_key}' not found in test metrics: {best_test_m.keys()}")
    
    return {
        "best_epoch_index": best_idx,
        "best_val_monitor": best_val,
        "train_metrics": best_train_m,
        "valid_metrics": best_valid_m,
        "test_metrics": best_test_m,
        "test_f1": f1_test,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to out_finetune pickle file, e.g. out_finetune/EEG/..._finetune",
    )
    parser.add_argument(
        "--metric_key",
        type=str,
        default="f1_score",
        help="Key for the F1 metric in the stored metric dict (default: f1_score)",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="accuracy",
        help="Validation metric used to select best epoch (default: accuracy)",
    )
    args = parser.parse_args()

    result = load_best_f1(args.file, metric_key=args.metric_key, monitor=args.monitor)

    print(f"Loaded file: {args.file}")
    print(f"Best epoch index: {result['best_epoch_index']}")
    print(f"Best validation {args.monitor}: {result['best_val_monitor']:.4f}")
    print("Best test metrics dict:", result["test_metrics"])
    print(f"Best test {args.metric_key}: {result['test_f1']}")


if __name__ == "__main__":
    main()
