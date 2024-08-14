import itertools
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import string

from pred_evals.globals import PERFORMANCE_METRICS_NICE_STRINGS_DICT
import pred_evals.plot
import pred_evals.utils


# refresh = True
refresh = False


data_dir, results_dir = pred_evals.utils.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

date_str = "2024-05-31"
results_dir = os.path.join(results_dir, date_str)
os.makedirs(results_dir, exist_ok=True)


sweep_ids = [
    "ix3lasrx",  # HellaSwag
    "jsonb7j8",  # PIQA
]

per_sample_correlations_between_scores_and_compute_df = (
    pred_evals.utils.download_wandb_project_runs_configs(
        wandb_project_path="predictable-llm-evals-compute-score-correlations",
        data_dir=data_dir,
        sweep_ids=sweep_ids,
        refresh=refresh,
        finished_only=False,
    )
)

# Keep only the runs that have data. Some combinations of benchmarks, subtasks and metrics don't exist.
per_sample_correlations_between_scores_and_compute_df = (
    per_sample_correlations_between_scores_and_compute_df[
        per_sample_correlations_between_scores_and_compute_df["no_data"] == False
    ]
)
per_sample_correlations_between_scores_and_compute_df.rename(
    columns={"model_family": "Model Family"}, inplace=True
)
per_sample_correlations_between_scores_and_compute_df[
    "Model Family"
] = per_sample_correlations_between_scores_and_compute_df["Model Family"].map(
    lambda k: pred_evals.globals.MODEL_FAMILIES_NICE_STRINGS_DICT.get(k, k)
)

sorted_unique_benchmark_and_optional_task = np.sort(
    per_sample_correlations_between_scores_and_compute_df[
        "benchmark_and_optional_task"
    ].unique()
)
sorted_unique_correlation_metrics = np.sort(
    per_sample_correlations_between_scores_and_compute_df["correlation_metric"].unique()
)
sorted_unique_metrics = [
    "$\\log p_{\\theta}^{\\text{Vocab}}(\\text{Correct Choice})$",
    "$p_{\\theta}^{\\text{Vocab}}(\\text{Correct Choice})$",
    "$p_{\\theta}^{\\text{Vocab}}(\\text{Correct Choice})$",
    "Accuracy",
]

# Define the continuous colormap
# Generate the discrete colors from the entire range of the continuous colormap
continuous_cmap = plt.get_cmap("coolwarm_r")
palette = sns.color_palette(
    continuous_cmap(
        np.linspace(0.15, 1.0, len(sorted_unique_metrics))
    )  # More blue than red.
)
sorted_unique_model_families = np.sort(
    per_sample_correlations_between_scores_and_compute_df["Model Family"].unique()
)

id_columns = [
    "benchmark_and_optional_task",
    "metric",
    "correlation_metric",
    "Model Family",
]
correlation_distribution_statistics_columns = [
    c
    for c in per_sample_correlations_between_scores_and_compute_df.columns.values
    if c.startswith("statistics_")
]

benchmark_metric_model_family_correlations_statistics_df = (
    per_sample_correlations_between_scores_and_compute_df[
        id_columns + correlation_distribution_statistics_columns
    ]
).copy()
benchmark_metric_model_family_correlations_statistics_df[
    "Model Family"
] = benchmark_metric_model_family_correlations_statistics_df["Model Family"].map(
    lambda k: pred_evals.globals.MODEL_FAMILIES_NICE_STRINGS_DICT.get(k, k)
)

benchmark_metric_model_family_correlations_statistics_df.sort_values(
    "benchmark_and_optional_task", inplace=True
)

ordered_unique_correlation_distribution_statistics = [
    "Mean(Correlations)",
    "Median(Correlations)",
    "AUC of Correlations' Complementary Cumulative Distribution Function",
    "-Min(Wasserstein(Correlations, 1), Wasserstein(Correlations, -1))",
]

correlation_distribution_statistics_results_dir = os.path.join(
    results_dir, "correlation_distribution_statistics"
)
os.makedirs(correlation_distribution_statistics_results_dir, exist_ok=True)

for (
    (correlation_metric,),
    benchmark_metric_model_family_statistics_df,
) in benchmark_metric_model_family_correlations_statistics_df.groupby(
    ["correlation_metric"]
):
    correlation_metric_results_dir = os.path.join(
        correlation_distribution_statistics_results_dir,
        f"correlation_metric={correlation_metric}",
    )
    os.makedirs(correlation_metric_results_dir, exist_ok=True)

    benchmark_metric_model_family_statistics_melted_df = (
        benchmark_metric_model_family_statistics_df.melt(
            id_vars=id_columns,
            value_vars=correlation_distribution_statistics_columns,
            var_name="Correlation Distribution Statistic",
            value_name="Statistic Value",
        )
    )
    benchmark_metric_model_family_statistics_melted_df[
        "Non-Nice-String Correlation Distribution Statistic"
    ] = benchmark_metric_model_family_statistics_melted_df[
        "Correlation Distribution Statistic"
    ]
    benchmark_metric_model_family_statistics_melted_df[
        "Correlation Distribution Statistic"
    ] = benchmark_metric_model_family_statistics_melted_df[
        "Non-Nice-String Correlation Distribution Statistic"
    ].map(
        lambda k: pred_evals.globals.CORRELATION_STATISTICS_NICE_STRINGS_DICT.get(k, k)
    )
    benchmark_metric_model_family_statistics_melted_df[
        "benchmark_and_optional_task"
    ] = benchmark_metric_model_family_statistics_melted_df[
        "benchmark_and_optional_task"
    ].map(
        lambda k: pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(k, k)
    )
    benchmark_metric_model_family_statistics_melted_df[
        "metric"
    ] = benchmark_metric_model_family_statistics_melted_df["metric"].map(
        lambda k: pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT.get(k, k)
    )
    benchmark_metric_model_family_statistics_melted_df.rename(
        columns={
            "Model Family": "Model Family",
            "metric": "Metric",
            "benchmark_and_optional_task": "Benchmark",
        },
        inplace=True,
    )

    # Discard NaN data.
    benchmark_metric_model_family_statistics_melted_df = (
        benchmark_metric_model_family_statistics_melted_df[
            ~benchmark_metric_model_family_statistics_melted_df[
                "Correlation Distribution Statistic"
            ].isna()
        ]
    )

    # Keep only the correlation distribution statistics that we want.
    benchmark_metric_model_family_statistics_melted_df = (
        benchmark_metric_model_family_statistics_melted_df[
            benchmark_metric_model_family_statistics_melted_df[
                "Correlation Distribution Statistic"
            ].isin(ordered_unique_correlation_distribution_statistics)
        ]
    )

    # Create continuous values for "Model Family"
    model_family_mapping = {
        model_family: i + 1
        for i, model_family in enumerate(
            benchmark_metric_model_family_statistics_melted_df["Model Family"].unique()
        )
    }
    benchmark_metric_model_family_statistics_melted_df[
        "Model Family Numeric"
    ] = benchmark_metric_model_family_statistics_melted_df["Model Family"].map(
        model_family_mapping
    )
    plt.close()
    g = sns.relplot(
        data=benchmark_metric_model_family_statistics_melted_df,
        kind="scatter",
        y="Model Family Numeric",
        hue="Benchmark",
        x="Statistic Value",
        col="Correlation Distribution Statistic",
        col_order=ordered_unique_correlation_distribution_statistics,
        row="Metric",
        row_order=sorted_unique_metrics,
        s=150,
        facet_kws={"margin_titles": True, "sharex": False, "sharey": True},
    )
    g.set(
        yticks=list(model_family_mapping.values()),
        yticklabels=list(model_family_mapping.keys()),
    )
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    pred_evals.plot.save_plot_with_multiple_extensions(
        plot_dir=correlation_metric_results_dir,
        plot_title=f"reviewer_P2sw_request_{correlation_metric}",
    )
    # plt.show()

print("Finished notebooks/07_neurips_dandb_reviewer_P2sw!")
