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
    "ged0642q",  # ARC (Challenge and Easy)
    "ix3lasrx",  # HellaSwag
    "pikdl9md",  # MathQA
    "cwtvun9t",  # MC TACO
    "kohac6et",  # MMLU (Part 0)
    "vhii07h7",  # MMLU (Part 1)
    "wn94qlma",  # MMLU (Part 2)
    "turw2dxt",  # MMLU (Part 3)
    "6jofdjdw",  # MMLU (Part 4)
    "nw4gyz7x",  # MMLU (Part 5)
    "osmch39v",  # MMLU (Part 6)
    "nr43unop",  # MMLU (Part 7)
    "t9sjmft4",  # MMLU (Part 8)
    "sa4soy9r",  # MMLU (Part 9)
    "iweoijp4",  # OpenbookQA
    "jsonb7j8",  # PIQA
    "5b4q2wu0",  # PubmedQA
    "386fswju",  # RACE
    "gv27y431",  # SciQ
    "x7656p1u",  # SocialIQA
    "vcdnekxs",  # TriviaQA
    "47p2h14r",  # Winogrande and xwinograd_en
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
    "log_prob_vocab_correct",
    "prob_vocab_correct",
    "prob_choices_correct",
    "brier_score",
    "acc",
    # "acc_norm",
    # "target_is_greedy",
]
# palette = sns.color_palette(palette="coolwarm_r", n_colors=len(sorted_unique_metrics))
# palette = sns.color_palette(palette="gnuplot2", n_colors=len(sorted_unique_metrics))
# palette = sns.color_palette(palette="inferno_r", n_colors=len(sorted_unique_metrics))
# palette = sns.color_palette(palette="rainbow_r", n_colors=len(sorted_unique_metrics))
# palette = sns.color_palette(palette="husl", n_colors=len(sorted_unique_metrics))
# palette = sns.color_palette("icefire", n_colors=len(sorted_unique_metrics))
# palette = sns.color_palette("Spectral", n_colors=len(sorted_unique_metrics))
# palette = sns.color_palette(palette="RdYlBu_r", n_colors=len(sorted_unique_metrics))
# palette = sns.color_palette(palette="cool_r", n_colors=len(sorted_unique_metrics))


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

correlation_distribution_statistics_results_dir = os.path.join(
    results_dir, "correlation_distribution_statistics"
)
os.makedirs(correlation_distribution_statistics_results_dir, exist_ok=True)

# for (
#     (correlation_metric,),
#     benchmark_metric_model_family_statistics_df,
# ) in benchmark_metric_model_family_correlations_statistics_df.groupby(
#     ["correlation_metric"]
# ):
#     correlation_metric_results_dir = os.path.join(
#         correlation_distribution_statistics_results_dir,
#         f"correlation_metric={correlation_metric}",
#     )
#     os.makedirs(correlation_metric_results_dir, exist_ok=True)
#
#     benchmark_metric_model_family_statistics_melted_df = (
#         benchmark_metric_model_family_statistics_df.melt(
#             id_vars=id_columns,
#             value_vars=correlation_distribution_statistics_columns,
#             var_name="Correlation Distribution Statistic",
#             value_name="Statistic Value",
#         )
#     )
#     benchmark_metric_model_family_statistics_melted_df[
#         "Non-Nice-String Correlation Distribution Statistic"
#     ] = benchmark_metric_model_family_statistics_melted_df[
#         "Correlation Distribution Statistic"
#     ]
#     benchmark_metric_model_family_statistics_melted_df[
#         "Correlation Distribution Statistic"
#     ] = benchmark_metric_model_family_statistics_melted_df[
#         "Non-Nice-String Correlation Distribution Statistic"
#     ].map(
#         lambda k: pred_evals.globals.CORRELATION_STATISTICS_NICE_STRINGS_DICT.get(k, k)
#     )
#     benchmark_metric_model_family_statistics_melted_df[
#         "benchmark_and_optional_task"
#     ] = benchmark_metric_model_family_statistics_melted_df[
#         "benchmark_and_optional_task"
#     ].map(
#         lambda k: pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(k, k)
#     )
#     benchmark_metric_model_family_statistics_melted_df[
#         "metric"
#     ] = benchmark_metric_model_family_statistics_melted_df["metric"].map(
#         lambda k: pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT.get(k, k)
#     )
#     benchmark_metric_model_family_statistics_melted_df.rename(
#         columns={"Model Family": "Model Family", "metric": "Metric"}, inplace=True
#     )
#
#     plt.close()
#     # palette = sns.color_palette("coolwarm_r", n_colors=len(sorted_unique_metrics))
#     # palette = sns.color_palette("icefire", n_colors=len(sorted_unique_metrics))
#     # palette = sns.color_palette("Spectral_r", n_colors=len(sorted_unique_metrics))
#     # palette = sns.color_palette(palette="RdYlBu_r", n_colors=len(sorted_unique_metrics))
#     # palette = sns.color_palette(palette="cool_r", n_colors=len(sorted_unique_metrics))
#     # palette = sns.color_palette(palette="seismic", n_colors=len(sorted_unique_metrics))
#     g = sns.relplot(
#         data=benchmark_metric_model_family_statistics_melted_df,
#         kind="scatter",
#         y="benchmark_and_optional_task",
#         x="Statistic Value",
#         hue="Metric",
#         hue_order=[
#             pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT.get(
#                 m,
#             )
#             for m in sorted_unique_metrics
#         ],
#         col="Correlation Distribution Statistic",
#         col_order=[
#             "Mean(Correlations)",
#             "Median(Correlations)",
#             "AUC of Correlations' Complementary Cumulative Distribution Function",
#             "-Min(Wasserstein(Correlations, 1), Wasserstein(Correlations, -1))",
#         ],
#         col_wrap=2,
#         style="Model Family",
#         style_order=sorted_unique_model_families,
#         s=250,
#         palette=palette,
#         linewidth=0,  # Remove white borders around points.
#         facet_kws={"sharex": False},
#         height=12,
#     )
#     g.set_axis_labels(y_var="Benchmark (and Optional Task)")
#     g.set_titles(col_template="")
#     # Iterate over each column and set the x-axis label.
#     # Setting x-axis labels based on column names
#     for ax, title in zip(g.axes.flat, g.col_names):
#         ax.set_xlabel(title)
#     g.fig.suptitle(
#         f"Correlation Distribution Statistics by Benchmark and Metric ({pred_evals.globals.CORRELATION_STATISTICS_NICE_STRINGS_DICT[correlation_metric]} Correlations)"
#     )
#     sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
#     pred_evals.plot.save_plot_with_multiple_extensions(
#         plot_dir=correlation_metric_results_dir,
#         plot_title=f"benchmark_vs_{correlation_metric}_correlation_distribution_statistics_by_model_family_by_metric",
#     )
#     # plt.show()
#     plt.close()
#
#     # Plot each subfigure separately.
#     for (
#         (statistic,),
#         benchmark_metric_model_family_statistic_melted_df,
#     ) in benchmark_metric_model_family_statistics_melted_df.groupby(
#         ["Non-Nice-String Correlation Distribution Statistic"]
#     ):
#         plt.close()
#         g = sns.relplot(
#             data=benchmark_metric_model_family_statistic_melted_df,
#             kind="scatter",
#             y="benchmark_and_optional_task",
#             x="Statistic Value",
#             hue="Metric",
#             hue_order=[
#                 pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT.get(
#                     m,
#                 )
#                 for m in sorted_unique_metrics
#             ],
#             style="Model Family",
#             style_order=sorted_unique_model_families,
#             s=250,
#             linewidth=0,  # Remove white borders around points.
#             height=12,
#             palette=palette,
#         )
#         g.set_axis_labels(
#             "Statistic Value",
#             "Benchmark (and Optional Task)",
#         )
#         g.fig.suptitle(
#             f"{pred_evals.globals.CORRELATION_STATISTICS_NICE_STRINGS_DICT.get(statistic, statistic)}\n"
#             f"{pred_evals.globals.CORRELATION_STATISTICS_NICE_STRINGS_DICT[correlation_metric]} Correlations"
#         )
#         sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
#         pred_evals.plot.save_plot_with_multiple_extensions(
#             plot_dir=correlation_metric_results_dir,
#             plot_title=f"benchmark_vs_{correlation_metric}_correlation_distribution_statistic={statistic}_by_model_family_by_metric",
#         )
#         # plt.show()
#         plt.close()
#
#
# for statistic_column in correlation_distribution_statistics_columns:
#     plt.close()
#     g = sns.relplot(
#         data=benchmark_metric_model_family_correlations_statistics_df,
#         kind="scatter",
#         y="benchmark_and_optional_task",
#         x=statistic_column,
#         hue="metric",
#         hue_order=sorted_unique_metrics,
#         style="Model Family",
#         style_order=sorted_unique_model_families,
#         col="correlation_metric",
#         col_order=["kendall", "pearson", "spearman"],
#         s=100,
#         linewidth=0,  # Remove white borders around points.
#         height=12,
#         palette=palette,
#     )
#     if statistic_column in pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT:
#         g.set(xlim=pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT[statistic_column])
#     if statistic_column in pred_evals.globals.CORRELATION_STATISTICS_NICE_STRINGS_DICT:
#         title = f"{pred_evals.globals.CORRELATION_STATISTICS_NICE_STRINGS_DICT[statistic_column]} by Benchmark and Metric"
#         xlabel = pred_evals.globals.CORRELATION_STATISTICS_NICE_STRINGS_DICT[
#             statistic_column
#         ]
#     else:
#         title = (f"{statistic_column} by Benchmark and Metric",)
#         xlabel = statistic_column
#     g.fig.suptitle(title)
#     g.set_axis_labels(
#         xlabel,
#         "Benchmark (and Optional Task)",
#     )
#     g.set_titles(col_template="Correlation: {col_name}")
#     sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
#     pred_evals.plot.save_plot_with_multiple_extensions(
#         plot_dir=correlation_distribution_statistics_results_dir,
#         plot_title=f"{statistic_column}_vs_benchmark",
#     )
#     # plt.show()
#     plt.close()

correlation_columns = [
    c
    for c in per_sample_correlations_between_scores_and_compute_df.columns.values
    if c.startswith("correlation_sample=")
]

per_sample_correlations_between_scores_and_compute_df = (
    per_sample_correlations_between_scores_and_compute_df[
        id_columns + correlation_columns
    ]
)


# Melt the dataframe so that each row is a sample.
# Columns: benchmark_and_optional_task, metric, correlation_metric, model_nicknames, sample_idx, correlation_score]
per_sample_correlations_between_scores_and_compute_df = (
    per_sample_correlations_between_scores_and_compute_df.melt(
        id_vars=id_columns,
        value_vars=correlation_columns,
        value_name="correlation_score",
        var_name="sample_idx",
    )
)

# Ensure that the correlation scores are floats.
per_sample_correlations_between_scores_and_compute_df[
    "correlation_score"
] = per_sample_correlations_between_scores_and_compute_df["correlation_score"].astype(
    float
)


# for (
#     (benchmark_and_optional_task, metric, correlation_metric),
#     benchmark_metric_correlation_df,
# ) in per_sample_correlations_between_scores_and_compute_df.groupby(
#     ["benchmark_and_optional_task", "metric", "correlation_metric"]
# ):
#     benchmark_and_metric_results_dir = os.path.join(
#         results_dir,
#         "correlation_distributions",
#         benchmark_and_optional_task,
#         metric,
#         correlation_metric,
#     )
#     os.makedirs(benchmark_and_metric_results_dir, exist_ok=True)
#
#     # Plot KDEs and ECDFs of correlation scores.
#     plt.close()
#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), sharey=False)
#     sns.kdeplot(
#         data=benchmark_metric_correlation_df,
#         x="correlation_score",
#         hue="Model Family",
#         hue_order=sorted_unique_model_families,
#         clip=(-1.0, 1.0),  # Don't evaluate KDE on impermissible correlations.
#         common_norm=False,
#         ax=axes[0],
#         legend=False,
#     )
#     axes[0].set_xlim(-1.0, 1.0)
#     axes[0].set_xlabel("Correlation Between FLOPs and Scores (Per Sample)")
#     axes[0].set_ylabel(
#         r"\% of Samples"
#     )  # Need to escape Latex's interpretation of % as a comment.
#
#     g = sns.ecdfplot(
#         data=benchmark_metric_correlation_df,
#         x="correlation_score",
#         hue="Model Family",
#         hue_order=sorted_unique_model_families,
#         complementary=True,
#         ax=axes[1],
#     )
#     axes[1].set_xlim(-1.0, 1.0)
#     axes[1].set_xlabel("Correlation Between FLOPs and Scores (Per Sample)")
#     axes[1].set_ylabel("1 - CDF")
#     sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
#     fig.suptitle(
#         f"Benchmark: {pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT[benchmark_and_optional_task]}\n"
#         + f"Performance Metric: {pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[metric]}\n"
#         + f"Correlation Metric: {pred_evals.globals.CORRELATION_STATISTICS_NICE_STRINGS_DICT[correlation_metric]}",
#     )
#
#     pred_evals.plot.save_plot_with_multiple_extensions(
#         plot_dir=benchmark_and_metric_results_dir,
#         plot_title=f"{benchmark_and_optional_task}_{metric}_kde_and_ecdf_vs_correlation_score_by_model_family_correlation_metric={correlation_metric}",
#     )
#     # plt.show()
#     plt.close()

# per_sample_correlations_between_scores_and_compute_df[
#     "sample_idx"
# ] = per_sample_correlations_between_scores_and_compute_df["sample_idx"].apply(
#     lambda s: int(s.split("=")[1])  # Convert "correlation_sample=0" to 0, etc.
# )
metrics_order = [
    "log_prob_vocab_correct",
    # "prob_vocab_correct",
    "prob_choices_correct",
    "brier_score",
    "acc",
    # "target_is_greedy",
]
performance_metrics_col_order = [
    pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[m] for m in metrics_order
]


for (
    benchmark_and_optional_task,
    benchmark_correlations_df,
) in per_sample_correlations_between_scores_and_compute_df.groupby(
    "benchmark_and_optional_task"
):
    benchmark_results_dir = os.path.join(
        results_dir, "correlation_distributions", benchmark_and_optional_task
    )
    os.makedirs(benchmark_results_dir, exist_ok=True)

    benchmark_correlations_df = benchmark_correlations_df.copy()
    benchmark_correlations_df["metric"] = benchmark_correlations_df["metric"].map(
        lambda m: pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT.get(m, m)
    )
    benchmark_correlations_df["Model Family"] = benchmark_correlations_df[
        "Model Family"
    ].map(lambda m: pred_evals.globals.MODEL_FAMILIES_NICE_STRINGS_DICT.get(m, m))
    benchmark_correlations_df.rename(
        columns={"Model Family": "Model Family"}, inplace=True
    )
    # Plot ECDFs of correlation scores.
    plt.close()
    plt.figure(figsize=(8, 8))
    g = sns.displot(
        data=benchmark_correlations_df[
            benchmark_correlations_df["correlation_metric"] == "spearman"
        ],
        kind="ecdf",
        complementary=True,
        x="correlation_score",
        hue="Model Family",
        hue_order=sorted_unique_model_families,
        # col="correlation_metric",
        col="metric",
        col_order=performance_metrics_col_order,
        col_wrap=2,
    )
    g.set(xlim=(-1.0, 1.0))
    g.fig.suptitle(
        f"{benchmark_and_optional_task}",
        y=1.02,
    )
    g.set_axis_labels("Correlation Between FLOPs and Scores (Per Sample)", "1 - CDF")
    g.set_titles(col_template="Metric: {col_name}")
    g.fig.suptitle(
        f"Distributions of Score-Compute Correlations by Metric\nBenchmark: {pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT[benchmark_and_optional_task]}"
    )
    # Add letters to each subplot.
    # See: https://stackoverflow.com/a/25544329/4570472
    for n, ax in enumerate(g.axes.flat):
        ax.text(
            -0.05,
            1.05,
            string.ascii_uppercase[n],
            transform=ax.transAxes,
            size=20,
            weight="bold",
        )
    # Move legend into the first axis
    legend = g._legend
    legend.set_bbox_to_anchor((0.05, 0.53))
    legend.set_loc(3)
    # Modify the legend to have a white background and border
    legend.set_frame_on(True)  # Enable the background frame
    legend.get_frame().set_facecolor("white")  # Set the background color to white
    legend.get_frame().set_edgecolor("black")  # Set the border color
    legend.get_frame().set_linewidth(1)  # Set the border width
    pred_evals.plot.save_plot_with_multiple_extensions(
        plot_dir=benchmark_results_dir,
        plot_title=f"{benchmark_and_optional_task}_ecdf_vs_correlation_score_by_model_family_split_performance_metric",
    )
    # plt.show()
    plt.close()


for (
    (
        benchmark_and_optional_task,
        metric,
    ),
    benchmark_metric_correlations_df,
) in per_sample_correlations_between_scores_and_compute_df.groupby(
    ["benchmark_and_optional_task", "metric"]
):
    benchmark_and_metric_results_dir = os.path.join(
        results_dir, "correlation_distributions", benchmark_and_optional_task, metric
    )
    os.makedirs(benchmark_and_metric_results_dir, exist_ok=True)

    # Plot KDEs of correlation scores.
    plt.close()
    g = sns.displot(
        data=benchmark_metric_correlations_df,
        kind="kde",
        x="correlation_score",
        hue="Model Family",
        hue_order=sorted_unique_model_families,
        col="correlation_metric",
        clip=(-1.0, 1.0),  # Don't evaluate KDE on impermissible correlations.
        common_norm=False,
    )
    g.set(xlim=(-1.0, 1.0))
    g.fig.suptitle(
        f"{pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT[benchmark_and_optional_task]}\n{pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[metric]}",
        y=1.02,
    )
    g.set_axis_labels("Correlation Between FLOPs and Scores (Per Sample)", "Density")
    g.set_titles(col_template="Correlation: {col_name}")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    pred_evals.plot.save_plot_with_multiple_extensions(
        plot_dir=benchmark_and_metric_results_dir,
        plot_title=f"{benchmark_and_optional_task}_{metric}_kde_vs_correlation_score_by_model_family_split_correlation_metric",
    )
    # plt.show()
    plt.close()

    # Plot ECDFs of correlation scores.
    plt.close()
    g = sns.displot(
        data=benchmark_metric_correlations_df,
        kind="ecdf",
        complementary=True,
        x="correlation_score",
        hue="Model Family",
        hue_order=sorted_unique_model_families,
        col="correlation_metric",
    )
    g.set(xlim=(-1.0, 1.0))
    g.fig.suptitle(
        f"{pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT[benchmark_and_optional_task]}\n{pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[metric]}",
        y=1.02,
    )
    g.set_axis_labels("Correlation Between FLOPs and Scores (Per Sample)", "1 - CDF")
    g.set_titles(col_template="Correlation: {col_name}")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    pred_evals.plot.save_plot_with_multiple_extensions(
        plot_dir=benchmark_and_metric_results_dir,
        plot_title=f"{benchmark_and_optional_task}_{metric}_ecdf_vs_correlation_score_by_model_family_split_correlation_metric",
    )
    # plt.show()
    plt.close()

    # TODO: Debug why there are duplicates for a single combination of benchmark_and_optional_task, metric, model_nicknames, and sample_idx.
    print(
        "Before dropping duplicates, benchmark_metric_correlations_df.shape:",
        benchmark_metric_correlations_df.shape,
    )
    benchmark_metric_correlations_df.drop_duplicates(inplace=True)
    print(
        "After dropping duplicates, benchmark_metric_correlations_df.shape:",
        benchmark_metric_correlations_df.shape,
    )
    benchmark_metric_correlations_df.reset_index(drop=True, inplace=True)

    benchmark_metric_correlations_pivoted_df = benchmark_metric_correlations_df.pivot(
        index=[
            "benchmark_and_optional_task",
            "metric",
            "Model Family",
            "sample_idx",
        ],
        columns="correlation_metric",
        values="correlation_score",
    )

    for corr_metr_1_idx in range(len(sorted_unique_correlation_metrics)):
        corr_metr_1 = sorted_unique_correlation_metrics[corr_metr_1_idx]
        for corr_metr_2_idx in range(
            corr_metr_1_idx + 1, len(sorted_unique_correlation_metrics)
        ):
            corr_metr_2 = sorted_unique_correlation_metrics[corr_metr_2_idx]
            plt.close()
            plt.figure(figsize=(10, 6))
            g = sns.scatterplot(
                data=benchmark_metric_correlations_pivoted_df,
                x=corr_metr_1,
                y=corr_metr_2,
                hue="Model Family",
                hue_order=sorted_unique_model_families,
            )
            g.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
            g.set_xlabel(
                f"{pred_evals.globals.CORRELATION_STATISTICS_NICE_STRINGS_DICT[corr_metr_1]} Correlation\nBetween FLOPs and Scores (Per Sample)",
            )
            g.set_ylabel(
                f"{pred_evals.globals.CORRELATION_STATISTICS_NICE_STRINGS_DICT[corr_metr_2]} Correlation\nBetween FLOPs and Scores (Per Sample)",
            )
            g.set_title(
                f"{pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT[benchmark_and_optional_task]}\n{pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[metric]}",
            )
            try:
                sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
            except ValueError:
                # We sometimes receive ValueError: Axes(0.125,0.11;0.775x0.77) has no legend attached.
                # Answer: Because some dataframes are entirely NaN e.g., GSM8k
                # TODO: Debug why
                pass
            pred_evals.plot.save_plot_with_multiple_extensions(
                plot_dir=benchmark_and_metric_results_dir,
                plot_title=f"{benchmark_and_optional_task}_{metric}_scatterplot_{corr_metr_2}_vs_{corr_metr_1}_by_model_family",
            )
            # plt.show()
            plt.close()

    benchmark_metric_correlations_pivoted_df = benchmark_metric_correlations_df.pivot(
        index=[
            "benchmark_and_optional_task",
            "metric",
            "correlation_metric",
            "sample_idx",
        ],
        columns="Model Family",
        values="correlation_score",
    ).reset_index()
    for model_fam_1_idx in range(len(sorted_unique_model_families)):
        model_fam_1 = sorted_unique_model_families[model_fam_1_idx]
        for model_fam_2_idx in range(
            model_fam_1_idx + 1, len(sorted_unique_model_families)
        ):
            model_fam_2 = sorted_unique_model_families[model_fam_2_idx]
            plt.close()
            plt.figure(figsize=(10, 6))
            try:
                g = sns.scatterplot(
                    data=benchmark_metric_correlations_pivoted_df,
                    x=model_fam_1,
                    y=model_fam_2,
                    hue="correlation_metric",
                )
            except ValueError:
                # Could not interpret value `INCITE_7B_PARAMETERS_TOKEN_FAMILY` for `y`. An entry with this name does not appear in `data`.
                continue

            g.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
            g.set_xlabel(f"Per-Sample Correlation(Compute, Score) for\n{model_fam_1}")
            g.set_ylabel(f"Per-Sample Correlation(Compute, Score) for\n{model_fam_2}")
            g.set_title(
                f"{pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT[benchmark_and_optional_task]}\n{pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[metric]}",
            )
            try:
                sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
            except ValueError:
                # We sometimes receive ValueError: Axes(0.125,0.11;0.775x0.77) has no legend attached.
                # Answer: Because some dataframes are entirely NaN e.g., GSM8k
                pass
            pred_evals.plot.save_plot_with_multiple_extensions(
                plot_dir=benchmark_and_metric_results_dir,
                plot_title=f"{benchmark_and_optional_task}_{metric}_scatterplot_{model_fam_2}_vs_{model_fam_1}_by_sample_idx",
            )
            # plt.show()
            plt.close()


print("Finished notebooks/03_correlations_between_sample_scores_and_compute!")
