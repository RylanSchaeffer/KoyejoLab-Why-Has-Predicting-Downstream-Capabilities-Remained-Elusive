import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
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


models_df = pred_evals.utils.load_models_df()
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

    # We only want specific metrics for this.
    all_model_all_benchmark_per_sample_results_df = (
        all_model_all_benchmark_per_sample_results_df[
            all_model_all_benchmark_per_sample_results_df["metric"].isin(
                {
                    "neg_log_prob_vocab_correct",
                    "log_prob_vocab_correct",
                    "prob_vocab_choices",
                    "prob_vocab_incorrect",
                    "prob_vocab_correct",
                }
            )
        ]
    )

    # Get rid of any models that aren't in the families of interest.
    all_model_all_benchmark_per_sample_results_df = (
        all_model_all_benchmark_per_sample_results_df[
            all_model_all_benchmark_per_sample_results_df["Model Nickname"].isin(
                set(model_nickname_to_model_family_dict.keys())
            )
        ]
    )

    # Add Model Family in.
    all_model_all_benchmark_per_sample_results_df[
        "Model Family"
    ] = all_model_all_benchmark_per_sample_results_df["Model Nickname"].map(
        model_nickname_to_model_family_dict
    )

    # Convert model familiy strings to nice model family strings.
    all_model_all_benchmark_per_sample_results_df[
        "Model Family"
    ] = all_model_all_benchmark_per_sample_results_df["Model Family"].map(
        lambda f: pred_evals.globals.MODEL_FAMILIES_NICE_STRINGS_DICT.get(f, f)
    )

    # TODO: Debug why there are a small number of duplicates.
    # Shape before dropping duplicates:  (47527858, 6)
    # Shape after dropping duplicates:   (47501478, 6)
    print(
        "Shape before dropping duplicates: ",
        all_model_all_benchmark_per_sample_results_df.shape,
    )
    all_model_all_benchmark_per_sample_results_df.drop_duplicates(inplace=True)
    print(
        "Shape after dropping duplicates: ",
        all_model_all_benchmark_per_sample_results_df.shape,
    )

    all_model_all_benchmark_per_sample_results_wide_df = (
        all_model_all_benchmark_per_sample_results_df.pivot(
            index=[
                "Model Family",
                "Model Nickname",
                "benchmark_and_optional_task",
                "sample_idx",
            ],
            columns="metric",
            values="score",
        ).reset_index()
    )

    all_model_all_benchmark_per_sample_results_wide_df[
        "neg_log_prob_vocab_choices"
    ] = -np.log(
        all_model_all_benchmark_per_sample_results_wide_df["prob_vocab_choices"]
    )

    all_model_all_benchmark_per_sample_results_wide_df.reset_index(
        drop=True
    ).to_feather(subsampled_data_path)
    del all_model_all_benchmark_per_sample_results_wide_df

all_model_all_benchmark_per_sample_results_wide_df = pd.read_feather(
    subsampled_data_path
)


# This will be the order of X and Y axes in the pairwise comparison plots.
metrics = [
    "Compute",
    "log_prob_vocab_correct",
    "prob_vocab_correct",
    "acc",
]

# Only visualize a few model families.
some_model_all_benchmark_per_sample_results_wide_df = (
    all_model_all_benchmark_per_sample_results_wide_df[
        all_model_all_benchmark_per_sample_results_wide_df["Model Family"].isin(
            [
                "Cerebras (Param. and Data Scaling)",
                "Pythia 12B Param. (Data Scaling)",
                "Pythia 300B Tokens (Param. Scaling)",
            ]
        )
    ].copy()
)

min_compute, max_compute = (
    # models_df["Compute"].min(),
    1e17,  # Manually override smallest_compute because it is too small otherwise.
    1.1 * models_df["Compute"].max(),  # Add a little allowance for visualizing.
)


results_per_benchmark_and_optional_task_dir = os.path.join(
    results_dir, "results_per_benchmark"
)
os.makedirs(results_per_benchmark_and_optional_task_dir, exist_ok=True)
for (
    benchmark_and_optional_task,
    some_model_one_benchmark_scores_wide_df,
) in some_model_all_benchmark_per_sample_results_wide_df.groupby(
    "benchmark_and_optional_task"
):
    print(
        "Generating plots for benchmark and optional task:", benchmark_and_optional_task
    )

    benchmark_and_optional_task_results_dir = os.path.join(
        results_dir, benchmark_and_optional_task
    )
    os.makedirs(benchmark_and_optional_task_results_dir, exist_ok=True)

    # Skip empty dataframes.
    if len(some_model_one_benchmark_scores_wide_df) == 0:
        print("Skipping because no data.")
        continue

    # Check that all 4 metrics are present. Otherwise, skip.
    if not all(
        metric in set(some_model_one_benchmark_scores_wide_df.columns.values)
        for metric in {
            "log_prob_vocab_correct",
            "prob_vocab_correct",
            "prob_vocab_incorrect",
        }
    ):
        print("Skipping because not all metrics are present.")
        continue

    # Add compute to the scores dataframe.
    some_model_one_benchmark_scores_wide_df = (
        some_model_one_benchmark_scores_wide_df.merge(
            models_df[["Model Nickname", "Compute"]],
            on="Model Nickname",
            how="left",
        )
    )

    # We want to make sure the X and Y axis are scaled the same way.
    smallest_x_quantile = np.nanquantile(
        some_model_one_benchmark_scores_wide_df["prob_vocab_correct"], 0.01
    )
    smallest_y_quantile = np.nanquantile(
        some_model_one_benchmark_scores_wide_df["prob_vocab_choices"], 0.01
    )
    min_value = min(smallest_x_quantile, smallest_y_quantile)
    min_value = max(min_value, 1e-50)  # Avoid log(0) errors.

    some_model_one_benchmark_scores_wide_df = some_model_one_benchmark_scores_wide_df[
        (
            some_model_one_benchmark_scores_wide_df["prob_vocab_correct"]
            >= smallest_x_quantile
        )
        & (
            some_model_one_benchmark_scores_wide_df["prob_vocab_incorrect"]
            >= smallest_y_quantile
        )
        & (some_model_one_benchmark_scores_wide_df["Compute"] >= min_compute)
        & (some_model_one_benchmark_scores_wide_df["Compute"] <= max_compute)
    ]

    if len(some_model_one_benchmark_scores_wide_df) == 0:
        print(
            "Skipping because no data after filtering based on probability_vocab_correct, probability_vocab_incorrect and compute."
        )
        continue

    some_model_one_benchmark_scores_wide_df["neg_log_prob_vocab_correct"] = -np.log(
        some_model_one_benchmark_scores_wide_df["prob_vocab_correct"]
    )
    some_model_one_benchmark_scores_wide_df["neg_log_prob_vocab_incorrect"] = -np.log(
        some_model_one_benchmark_scores_wide_df["prob_vocab_incorrect"]
    )

    # Melt based on which probability we're considering (correct vs. other choices).
    some_model_one_benchmark_neg_log_likelihoods_wide_df = (
        some_model_one_benchmark_scores_wide_df.melt(
            id_vars=[
                "Model Family",
                "Model Nickname",
                "benchmark_and_optional_task",
                "sample_idx",
                "Compute",
            ],
            value_vars=[
                "neg_log_prob_vocab_correct",
                "neg_log_prob_vocab_incorrect",
            ],
            var_name="Cross Entropy Type",
            value_name="cross_entropy",
        )
    )
    some_model_one_benchmark_neg_log_likelihoods_wide_df[
        "Probability Type"
    ] = some_model_one_benchmark_neg_log_likelihoods_wide_df["Cross Entropy Type"].map(
        {
            "neg_log_prob_vocab_correct": pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[
                "neg_log_prob_vocab_correct"
            ],
            "neg_log_prob_vocab_incorrect": pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[
                "neg_log_prob_vocab_incorrect"
            ],
        }
    )
    prob_type_order = [
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[
            "neg_log_prob_vocab_correct"
        ],
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[
            "neg_log_prob_vocab_incorrect"
        ],
    ]

    plt.close()
    plt.figure(figsize=(10, 6))
    g = sns.relplot(
        data=some_model_one_benchmark_neg_log_likelihoods_wide_df,
        kind="line",
        x="Compute",
        y="cross_entropy",
        hue="Model Family",
        col="Probability Type",
        col_order=prob_type_order,
        err_style="bars",
        errorbar=("se", 2),
        # style="Probability Type",
        # style_order=prob_type_order,
    )
    g.set(xscale="log", yscale="log")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    g.set_axis_labels(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["Compute"],
        "Cross Entropy",
    )
    g.set_titles(
        col_template="{col_name}",
    )
    g.fig.suptitle(
        pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(
            benchmark_and_optional_task, benchmark_and_optional_task
        )
    )
    g.fig.subplots_adjust(top=0.9)
    pred_evals.plot.save_plot_with_multiple_extensions(
        plot_dir=benchmark_and_optional_task_results_dir,
        plot_title=f"{benchmark_and_optional_task}_neg_log_prob_other_choices_vocab_log_vs_compute_log_by_model_family",
    )
    # plt.show()

    # Melt based on which probability we're considering (correct vs. other choices).
    some_model_one_benchmark_prob_scores_melted_df = (
        some_model_one_benchmark_scores_wide_df.melt(
            id_vars=[
                "Model Family",
                "Model Nickname",
                "benchmark_and_optional_task",
                "sample_idx",
                "Compute",
            ],
            value_vars=[
                "prob_vocab_correct",
                "prob_vocab_incorrect",
            ],
            var_name="Probability Type",
            value_name="probability_mass",
        )
    )
    some_model_one_benchmark_prob_scores_melted_df[
        "Probability Type"
    ] = some_model_one_benchmark_prob_scores_melted_df["Probability Type"].map(
        {
            "prob_vocab_correct": pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[
                "prob_vocab_correct"
            ],
            "prob_vocab_incorrect": pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[
                "prob_vocab_incorrect"
            ],
        }
    )
    prob_type_order = [
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_vocab_correct"],
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[
            "prob_vocab_incorrect"
        ],
    ]

    plt.close()
    plt.figure(figsize=(10, 6))
    g = sns.relplot(
        data=some_model_one_benchmark_prob_scores_melted_df,
        kind="line",
        x="Compute",
        y="probability_mass",
        hue="Model Family",
        col="Probability Type",
        col_order=prob_type_order,
        err_style="bars",
        errorbar=("se", 2),
        # style="Probability Type",
        # style_order=prob_type_order,
    )
    min_y_value = max(
        some_model_one_benchmark_prob_scores_melted_df.groupby(
            ["Model Family", "Probability Type", "Compute"]
        )["probability_mass"]
        .mean()
        .min()
        / 5.0,  # 5.0 is a heuristic for visibility.
        1e-16,
    )
    g.set(xscale="log", yscale="log", ylim=(min_y_value, None))
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    g.set_axis_labels(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["Compute"],
        "Probability Mass",
    )
    g.set_titles(
        col_template="{col_name}",
    )
    g.fig.suptitle(
        pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(
            benchmark_and_optional_task, benchmark_and_optional_task
        )
    )
    g.fig.subplots_adjust(top=0.9)
    pred_evals.plot.save_plot_with_multiple_extensions(
        plot_dir=benchmark_and_optional_task_results_dir,
        plot_title=f"{benchmark_and_optional_task}_prob_other_choices_vocab_log_vs_compute_log_by_model_family",
    )
    # plt.show()

    plt.close()
    g = sns.relplot(
        data=some_model_one_benchmark_scores_wide_df,
        kind="scatter",
        x="prob_vocab_correct",
        y="prob_vocab_incorrect",
        hue="Compute",
        hue_norm=matplotlib.colors.LogNorm(vmin=min_compute, vmax=max_compute),
        col="Model Family",
        s=10,
        linewidth=0,
        # palette="copper",
        palette="mako",
        facet_kws={"sharex": False, "sharey": False},
    )
    g.set(
        xlim=(min_value, 1.0),
        ylim=(min_value, 1.0),
        xscale="log",
        yscale="log",
    )
    # g.set_titles(col_template="Correlation: {col_name}")
    g.set_axis_labels(
        x_var=pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[
            "prob_vocab_correct"
        ],
        y_var=pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[
            "prob_vocab_incorrect"
        ],
    )
    g.set_titles(col_template="Model Family: {col_name}")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    g.fig.suptitle(
        pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(
            benchmark_and_optional_task, benchmark_and_optional_task
        )
    )
    g.fig.subplots_adjust(top=0.9)
    pred_evals.plot.save_plot_with_multiple_extensions(
        plot_dir=benchmark_and_optional_task_results_dir,
        plot_title=f"{benchmark_and_optional_task}_prob_other_choices_vocab_vs_prob_correct_vocab_scatter_by_compute_split_model_family",
    )
    # plt.show()


print("Finished notebooks/05_pvocab_vs_pcorrect_masses_fluctuations!")
