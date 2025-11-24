import numpy as np
from sklearn.metrics import recall_score, precision_score

np.random.seed(42)


def bootstrap_recall_ci_multi(y_true_list, y_pred_list, B=1000, alpha=0.05):
    """
    Calculate the bootstrap confidence interval and mean for recall across multiple experiments.

    Args:
        y_true_list (list of np.ndarray): List of true label arrays for each experiment.
        y_pred_list (list of np.ndarray): List of predicted label arrays for each experiment.
        B (int): Number of bootstrap resamplings per experiment (default: 1000).
        alpha (float): Significance level (default: 0.05).

    Returns:
        tuple: (ci_lower, ci_upper)
    """
    assert len(y_true_list) == len(
        y_pred_list
    ), "y_true_list and y_pred_list must have the same length!"

    all_bootstrap_recalls = []

    for y_true, y_pred in zip(y_true_list, y_pred_list):
        for _ in range(B):
            idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
            y_true_b = y_true[idx]
            y_pred_b = y_pred[idx]
            recall = recall_score(y_true_b, y_pred_b)
            all_bootstrap_recalls.append(recall)

    all_bootstrap_recalls = np.array(all_bootstrap_recalls)

    ci_lower = np.percentile(all_bootstrap_recalls, 100 * (alpha / 2))
    ci_upper = np.percentile(all_bootstrap_recalls, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper


def bootstrap_precision_ci_multi(y_true_list, y_pred_list, B=1000, alpha=0.05):
    """
    Calculate the bootstrap confidence interval and mean for precision across multiple experiments.

    Args:
        y_true_list (list of np.ndarray): List of true label arrays for each experiment.
        y_pred_list (list of np.ndarray): List of predicted label arrays for each experiment.
        B (int): Number of bootstrap resamplings per experiment (default: 1000).
        alpha (float): Significance level (default: 0.05).

    Returns:
        tuple: (ci_lower, ci_upper)
    """
    assert len(y_true_list) == len(
        y_pred_list
    ), "y_true_list and y_pred_list must have the same length!"

    all_bootstrap_precisions = []

    for y_true, y_pred in zip(y_true_list, y_pred_list):
        for _ in range(B):
            idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
            y_true_b = y_true[idx]
            y_pred_b = y_pred[idx]
            precision = precision_score(y_true_b, y_pred_b)
            all_bootstrap_precisions.append(precision)

    all_bootstrap_precisions = np.array(all_bootstrap_precisions)

    ci_lower = np.percentile(all_bootstrap_precisions, 100 * (alpha / 2))
    ci_upper = np.percentile(all_bootstrap_precisions, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper


import numpy as np


def generate_labels_from_statistics(
    TP, real_positive, predicted_positive, total_samples
):
    """
    Generate y_true and y_pred arrays based on TP, TP+FN, TP+FP, and total sample count.

    Args:
        TP (int): True Positives
        real_positive (int): TP + FN (number of real/actual positives)
        predicted_positive (int): TP + FP (number of predicted positives)
        total_samples (int): Total number of samples

    Returns:
        tuple: y_true (np.ndarray), y_pred (np.ndarray)
    """
    # Derive other metrics
    FN = real_positive - TP
    FP = predicted_positive - TP
    TN = total_samples - (TP + FP + FN)

    if TN < 0:
        raise ValueError(
            "Invalid parameter combination: TN is negative, please check your inputs."
        )

    # Build y_true
    y_true = np.array(
        [1] * TP  # True positives
        + [1] * FN  # False negatives
        + [0] * FP  # False positives
        + [0] * TN  # True negatives
    )

    # Build y_pred
    y_pred = np.array(
        [1] * TP  # True positives
        + [0] * FN  # False negatives
        + [1] * FP  # False positives
        + [0] * TN  # True negatives
    )

    # Shuffle the arrays
    indices = np.arange(len(y_true))
    np.random.shuffle(indices)

    return y_true[indices], y_pred[indices]


# Example usage (according to your initial data: TP=7, TP+FN=7, TP+FP=20, total=40)
# y_true, y_pred = generate_labels_from_statistics(TP=7, real_positive=7, predicted_positive=20, total_samples=40)
