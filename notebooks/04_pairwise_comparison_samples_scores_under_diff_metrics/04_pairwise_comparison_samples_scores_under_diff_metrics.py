import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from typing import List

import pred_evals.globals
import pred_evals.metrics
import pred_evals.plot
import pred_evals.utils


data_dir, results_dir = pred_evals.utils.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)
# date_str = "2024-04-23"
date_str = "2024-05-31"
results_dir = os.path.join(results_dir, date_str)
os.makedirs(results_dir, exist_ok=True)


models_and_compute_df = pred_evals.utils.load_models_df()

min_compute, max_compute = (
    # models_df["Compute"].min(),
    1e17,  # Manually override smallest_compute because it is too small otherwise.
    1.1
    * models_and_compute_df["Compute"].max(),  # Add a little allowance for visualizing.
)
benchmarks_df = pd.read_csv("configs/benchmarks.csv", index_col=None)


model_families_of_interest = [
    "CEREBRAS_ALL_FAMILY",
    "INCITE_7B_PARAMETERS_TOKEN_FAMILY",
    "LLM360_AMBER_7B_TOKENS_FAMILY",
    "OLMo_7B_PARAMETERS_TOKENS_FAMILY",
    "PYTHIA_12B_PARAMETERS_TOKENS_FAMILY",
    "PYTHIA_300B_TOKENS_PARAMETERS_FAMILY",
]

# Add the model family to the dataframe, based on Model Nickname.
model_nickname_to_model_family_dict = {
    model_name: model_family
    for model_family in model_families_of_interest
    for model_name in getattr(pred_evals.globals, model_family)
}

# Only keep rows in the desired model families.
# Cache to speed up iteration time.
subsampled_data_path = os.path.join(
    data_dir, f"subsampled_all_model_benchmark_per_sample_results_{date_str}.feather"
)
if not os.path.exists(subsampled_data_path):
    all_model_all_benchmark_per_sample_results_df = pd.read_feather(
        f"eval_results/all_model_benchmark_per_sample_results_{date_str}.feather"
    )
    all_model_all_benchmark_per_sample_results_df = (
        all_model_all_benchmark_per_sample_results_df[
            all_model_all_benchmark_per_sample_results_df["Model Nickname"].isin(
                set(model_nickname_to_model_family_dict.keys())
            )
        ]
    )
    all_model_all_benchmark_per_sample_results_df.reset_index(drop=True).to_feather(
        subsampled_data_path
    )
    del all_model_all_benchmark_per_sample_results_df

all_model_all_benchmark_per_sample_results_df = pd.read_feather(subsampled_data_path)

# We will join "Compute" in subsequently.
metrics_we_want = {
    "log_prob_vocab_correct",
    "prob_vocab_correct",
    "prob_choices_correct",
    "acc",
    "brier_score",
    "target_is_greedy",
}

results_per_benchmark_and_optional_task_dir = os.path.join(
    results_dir, "results_per_benchmark"
)
os.makedirs(results_per_benchmark_and_optional_task_dir, exist_ok=True)
for (
    benchmark_and_optional_task,
    benchmark_and_optional_task_scores_df,
) in tqdm(
    all_model_all_benchmark_per_sample_results_df.groupby("benchmark_and_optional_task")
):
    # Don't want to look at these now. Skip to save time.
    if benchmark_and_optional_task.startswith("advanced_ai_risk"):
        continue
    elif benchmark_and_optional_task.startswith("persona"):
        continue

    # Check that all 4 metrics are present. Otherwise, skip.
    unique_metrics = set(benchmark_and_optional_task_scores_df["metric"].unique())
    if not metrics_we_want.issubset(unique_metrics):
        continue

    benchmark_results_dir = os.path.join(results_dir, benchmark_and_optional_task)
    os.makedirs(benchmark_results_dir, exist_ok=True)

    benchmark_and_optional_task_scores_wide_df = (
        benchmark_and_optional_task_scores_df.pivot(
            index=[
                "Model Nickname",
                "benchmark_and_optional_task",
                "sample_idx",
            ],
            columns="metric",
            values="score",
        ).reset_index()
    )

    # Add in model families.
    benchmark_and_optional_task_scores_wide_df[
        "Model Family"
    ] = benchmark_and_optional_task_scores_wide_df["Model Nickname"].map(
        model_nickname_to_model_family_dict
    )
    benchmark_and_optional_task_scores_wide_df[
        "Model Family"
    ] = benchmark_and_optional_task_scores_wide_df["Model Family"].map(
        lambda m: pred_evals.globals.MODEL_FAMILIES_NICE_STRINGS_DICT.get(m, m)
    )

    # Add compute the scores dataframe.
    benchmark_and_optional_task_scores_wide_df = (
        benchmark_and_optional_task_scores_wide_df.merge(
            models_and_compute_df[["Model Nickname", "Compute"]],
            on="Model Nickname",
            how="left",
        )
    )

    # Plot two KDEs side-by-side: one for prob_choices_correct vs prob_vocab_correct,
    # and one for acc vs prob_choices_correct.
    # TODO: Debug why the log scaling makes the KDEs look so bad.
    # pred_evals.plot.plot_prob_choices_correct_vs_prob_vocab_correct_and_acc_vs_prob_choices_correct_by_model_family_kde(
    #     benchmark_and_optional_task_scores_wide_df=benchmark_and_optional_task_scores_wide_df,
    #     benchmark_and_optional_task=benchmark_and_optional_task,
    #     plot_dir=benchmark_results_dir,
    # )

    # Plot two scatter plots side-by-side: one for prob_choices_correct vs prob_vocab_correct,
    # and one for acc vs prob_choices_correct.
    pred_evals.plot.plot_prob_choices_correct_vs_prob_vocab_correct_and_acc_vs_prob_choices_correct_by_model_family_scatter(
        benchmark_and_optional_task_scores_wide_df=benchmark_and_optional_task_scores_wide_df,
        benchmark_and_optional_task=benchmark_and_optional_task,
        plot_dir=benchmark_results_dir,
    )

    pred_evals.plot.plot_prob_choices_correct_vs_prob_vocab_correct_and_acc_vs_prob_choices_correct_and_brier_score_vs_prob_choices_correct_by_model_family_scatter(
        benchmark_and_optional_task_scores_wide_df=benchmark_and_optional_task_scores_wide_df,
        benchmark_and_optional_task=benchmark_and_optional_task,
        plot_dir=benchmark_results_dir,
    )

    pred_evals.plot.plot_prob_choices_correct_vs_prob_vocab_correct_and_acc_vs_prob_vocab_correct_by_model_family_scatter(
        benchmark_and_optional_task_scores_wide_df=benchmark_and_optional_task_scores_wide_df,
        benchmark_and_optional_task=benchmark_and_optional_task,
        plot_dir=benchmark_results_dir,
    )

    pred_evals.plot.plot_prob_choices_correct_vs_prob_vocab_correct_and_brier_score_vs_prob_vocab_correct_and_acc_vs_prob_vocab_correct_by_model_family_scatter(
        benchmark_and_optional_task_scores_wide_df=benchmark_and_optional_task_scores_wide_df,
        benchmark_and_optional_task=benchmark_and_optional_task,
        plot_dir=benchmark_results_dir,
    )

    pred_evals.plot.plot_brier_score_vs_prob_vocab_correct_by_model_family(
        benchmark_and_optional_task_scores_wide_df=benchmark_and_optional_task_scores_wide_df,
        benchmark_and_optional_task=benchmark_and_optional_task,
        plot_dir=benchmark_results_dir,
    )

    pred_evals.plot.plot_target_is_greedy_vs_prob_choices_correct_by_model_family(
        benchmark_and_optional_task_scores_wide_df=benchmark_and_optional_task_scores_wide_df,
        benchmark_and_optional_task=benchmark_and_optional_task,
        plot_dir=benchmark_results_dir,
    )

    pred_evals.plot.plot_target_is_greedy_vs_prob_vocab_correct_by_model_family(
        benchmark_and_optional_task_scores_wide_df=benchmark_and_optional_task_scores_wide_df,
        benchmark_and_optional_task=benchmark_and_optional_task,
        plot_dir=benchmark_results_dir,
    )


print("Finished notebooks/04_pairwise_comparison_samples_scores_under_diff_metrics!")
