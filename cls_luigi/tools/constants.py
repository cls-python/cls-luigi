from dataclasses import dataclass
from typing import List

MINIMIZE = 1
MAXIMIZE = -1


# todo review these metrics
class MLMAXIMIZATIONMETRICS:
    metrics: List[str] = [
        'accuracy',
        'balanced_accuracy',
        'top_k_accuracy',
        'average_precision',
        'f1',
        'f1_micro',
        'f1_macro',
        'f1_weighted',
        'f1_samples',
        'precision',
        'recall',
        'jaccard',
        'roc_auc',
        'roc_auc_ovr',
        'roc_auc_ovo',
        'roc_auc_ovr_weighted',
        'roc_auc_ovo_weighted',
        'adjusted_mutual_info_score',
        'adjusted_rand_score',
        'completeness_score',
        'fowlkes_mallows_score',
        'homogeneity_score',
        'mutual_info_score',
        'normalized_mutual_info_score',
        'rand_score',
        'v_measure_score',
        'explained_variance',
        'r2',
        'd2_absolute_error_score'
    ]


# todo review these metrics
class MLMINIMIZATIONMETRICS:
    metrics: List[str] = [
        'neg_brier_score',  # Brier score is minimized but neg_* means it should be maximized numerically
        'neg_log_loss',
        'neg_mean_absolute_error',
        'neg_mean_squared_error',
        'neg_root_mean_squared_error',
        'neg_mean_squared_log_error',
        'neg_root_mean_squared_log_error',
        'neg_median_absolute_error',
        'max_error',
        'neg_mean_poisson_deviance',
        'neg_mean_gamma_deviance',
        'neg_mean_absolute_percentage_error',
        'd2_log_loss_score'
    ]


if __name__ == "__main__":
    print(MLMAXIMIZATIONMETRICS.metrics)
    print(MLMINIMIZATIONMETRICS.metrics)
    print(MINIMIZE)
    print(MAXIMIZE)