import os

# Rok asked us to include the following specifications in our code to prevent CPUs from spinning idly:
n_threads_str = "4"
os.environ["OMP_NUM_THREADS"] = n_threads_str
os.environ["OPENBLAS_NUM_THREADS"] = n_threads_str
os.environ["MKL_NUM_THREADS"] = n_threads_str
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads_str
os.environ["NUMEXPR_NUM_THREADS"] = n_threads_str

from builtins import getattr
import numpy as np
import pandas as pd
import pprint
import scipy.stats
import sklearn.metrics
from typing import Any, Dict, List
import wandb

import pred_evals.globals
import pred_evals.utils

run = wandb.init(
    project="predictable-llm-evals-compute-score-correlations",
    config=pred_evals.globals.default_correlations_between_sample_scores_and_compute_config,
    entity=pred_evals.utils.retrieve_wandb_username(),
)
wandb_config: Dict[str, Any] = dict(wandb.config)

# Convert from the model family string to the actual models.
wandb_config["model_nicknames"]: List[str] = getattr(
    pred_evals.globals, wandb_config["model_family"]
)
wandb.config.update({"model_nicknames": str(wandb_config["model_nicknames"])})

pprint.pprint(wandb_config)

# Load the dataframe of all results.
all_model_all_benchmark_per_sample_results_df = pd.read_feather(
    f"eval_results/all_model_benchmark_per_sample_results_{wandb_config['evals_collation_date']}.feather"
)

# Slice the model and benchmark per-sample scores.
model_benchmark_per_sample_results_tall_df = (
    all_model_all_benchmark_per_sample_results_df[
        (
            all_model_all_benchmark_per_sample_results_df["benchmark_and_optional_task"]
            == wandb_config["benchmark_and_optional_task"]
        )
        & (
            all_model_all_benchmark_per_sample_results_df["Model Nickname"].isin(
                wandb_config["model_nicknames"]
            )
        )
        & (
            all_model_all_benchmark_per_sample_results_df["metric"]
            == wandb_config["metric"]
        )
    ]
)

model_benchmark_per_sample_results_wide_df = (
    model_benchmark_per_sample_results_tall_df.pivot(
        index="Model Nickname",
        columns="sample_idx",
        values="score",
    )
)

# If there is no data, then log so that we can exclude from analyses.
if len(model_benchmark_per_sample_results_wide_df) == 0:
    wandb.log({"no_data": True})
    exit(0)
else:
    wandb.log({"no_data": False})


# We will also need the compute (FLOPS) for each model. Let's load that.
models_and_compute_df: pd.DataFrame = pred_evals.utils.load_models_df()

# Take only the models that we are interested in.
models_and_compute_df = models_and_compute_df[
    models_and_compute_df["Model Nickname"].isin(wandb_config["model_nicknames"])
]

# Add additional compute variables.
models_and_compute_df = pred_evals.utils.add_additional_compute_columns(
    df=models_and_compute_df
)

# Ensure that rows are sorted based on compute and are consistent.
sort_indices = np.argsort(models_and_compute_df["Compute"])
models_and_compute_df = models_and_compute_df.iloc[sort_indices]
model_benchmark_per_sample_results_wide_df = (
    model_benchmark_per_sample_results_wide_df.loc[
        models_and_compute_df["Model Nickname"]
    ]
)

# Shape: (n_models, n_samples)
scores = model_benchmark_per_sample_results_wide_df.values
n_models, n_samples = scores.shape
# Shape: (n_models,)
compute = models_and_compute_df["Log10 Compute"].values
assert len(compute) == n_models

# Brier score is a loss, so we negate it to make it a positive score.
if wandb_config["metric"] in {"brier_score"}:
    scores = -scores

# If a model had literally 0 FLOPS, then Log10(0) is not finite.
# We exclude these points because certain correlation functions can't handle NaN values.
valid_indices = np.isfinite(compute)
compute = compute[valid_indices]
scores = scores[valid_indices, :]


if wandb_config["correlation_metric"] == "kendall":
    correlation_func = scipy.stats.kendalltau
elif wandb_config["correlation_metric"] == "pearson":
    correlation_func = scipy.stats.pearsonr
elif wandb_config["correlation_metric"] == "spearman":
    correlation_func = scipy.stats.spearmanr
else:
    raise ValueError(
        f"Invalid correlation metric: {wandb_config['correlation_metric']}"
    )

# For each sample in the scores, compute how correlated it is with compute.
correlations = np.full(shape=n_samples, fill_value=np.nan)
p_values = np.full(shape=n_samples, fill_value=np.nan)
for sample_idx in range(n_samples):
    sample_scores = scores[:, sample_idx]
    correlation, p_value = correlation_func(compute, sample_scores)
    correlations[sample_idx] = correlation
    p_values[sample_idx] = p_value
    print(f"Sample {sample_idx}: correlation={correlation}, p_value={p_value}")


non_nan_correlations = correlations[np.isfinite(correlations)]
# ValueError: `sample` must not contain nan.
correlations_ecdf_result = scipy.stats.ecdf(non_nan_correlations)

# import matplotlib.pyplot as plt
#
# ax = plt.subplot()
# correlations_ecdf_result.sf.plot(ax)
# plt.xlim(-1.0, 1.0)
# plt.xlabel("Correlation")
# plt.ylabel("1 - ECDF")
# plt.show()

quantiles = correlations_ecdf_result.sf.quantiles
survival_values = correlations_ecdf_result.sf.probabilities

# Ensure that the edge quantities are correct for survival function.
# Without this, AUC will compute incorrect values.
quantiles = np.concatenate(
    [[-1.0000000000000000001], quantiles, [1.0000000000000000001]]
)
survival_values = np.concatenate([[1.0], survival_values, [0.0]])

correlations_auc = sklearn.metrics.auc(x=quantiles, y=survival_values)
# The largest AUC can be is 2.0 because x in [-1, 1] and survival function is between 0 and 1.
statistics_correlation_auc_max = max(correlations_auc, 2.0 - correlations_auc)


ideal_dist_all_samples_correlated = np.full_like(correlations, fill_value=1.0)
ideal_dist_all_samples_anticorrelated = np.full_like(correlations, fill_value=-1.0)

# Compute the Wasserstein distance between the correlations and the two ideal distributions.
wasserstein_distance_1 = scipy.stats.wasserstein_distance(
    non_nan_correlations,
    ideal_dist_all_samples_correlated,
)
wasserstein_distance_2 = scipy.stats.wasserstein_distance(
    non_nan_correlations,
    ideal_dist_all_samples_anticorrelated,
)
wasserstein_distance_min = min(wasserstein_distance_1, wasserstein_distance_2)

energy_distance_1 = scipy.stats.energy_distance(
    non_nan_correlations, ideal_dist_all_samples_correlated
)
energy_distance_2 = scipy.stats.energy_distance(
    non_nan_correlations, ideal_dist_all_samples_anticorrelated
)
energy_distance_min = min(energy_distance_1, energy_distance_2)

kolmogorov_smirnov_1, _ = scipy.stats.kstest(
    non_nan_correlations, ideal_dist_all_samples_correlated
)
kolmogorov_smirnov_2, _ = scipy.stats.kstest(
    non_nan_correlations, ideal_dist_all_samples_anticorrelated
)
kolmogorov_smirnov_min = min(kolmogorov_smirnov_1, kolmogorov_smirnov_2)

correlation_distribution_statistics_dict = {
    "statistics_correlation_auc": correlations_auc,
    "statistics_correlation_auc_max": statistics_correlation_auc_max,
    "statistics_neg_wasserstein_distance_1": -wasserstein_distance_1,
    "statistics_neg_wasserstein_distance_2": -wasserstein_distance_2,
    "statistics_neg_wasserstein_distance_min": -wasserstein_distance_min,
    "statistics_neg_energy_distance_1": -energy_distance_1,
    "statistics_neg_energy_distance_2": -energy_distance_2,
    "statistics_neg_energy_distance_min": -energy_distance_min,
    "kolmogorov_smirnov_1": kolmogorov_smirnov_1,
    "kolmogorov_smirnov_2": kolmogorov_smirnov_2,
    "kolmogorov_smirnov_min": kolmogorov_smirnov_min,
    "statistics_correlation_mean": np.mean(non_nan_correlations),
    "statistics_correlation_median": np.median(non_nan_correlations),
    "statistics_correlation_stddev": np.std(non_nan_correlations),
    "statistics_correlation_var": np.var(non_nan_correlations),
    "statistics_correlation_skew": scipy.stats.skew(non_nan_correlations),
    "statistics_correlation_kurtosis": scipy.stats.kurtosis(non_nan_correlations),
    "statistics_correlation_frac_nan": np.mean(np.isnan(correlations)),
    "statistics_correlation_num_nan": np.sum(np.isnan(correlations)),
}


# Log the correlations and p-values.
correlations_per_sample_dict = {
    f"correlation_sample={sample_idx}": correlation
    for sample_idx, correlation in enumerate(correlations)
}
p_values_dict = {
    f"p_value_sample={sample_idx}": p_value
    for sample_idx, p_value in enumerate(p_values)
}

all_results = {
    **correlation_distribution_statistics_dict,
    **correlations_per_sample_dict,
    **p_values_dict,
}

wandb.log(all_results)

# Explicitly tell W&B that training has finished.
wandb.finish()
