import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

import pred_evals.globals
import pred_evals.metrics

# Enable LaTeX rendering.
# https://stackoverflow.com/a/23856968
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage {foo - name}`...')
plt.rcParams["text.usetex"] = True
preamble_commands = [r"\usepackage{amsmath}"]  # Can add more commands to this list
plt.rcParams["text.latex.preamble"] = "\n".join(preamble_commands)
# Increase font size.
plt.rcParams.update({"font.size": 20})

sns.set_style("whitegrid")

sns.set_theme(style="whitegrid")


def plot_avg_score_vs_sample_idx_split_metric(
    benchmark_and_optional_task_scores_df: pd.DataFrame,
    benchmark_and_optional_task: str,
    plot_dir: str,
    plot_title: str = "avg_score_vs_sample_idx",
):
    for row_idx, (
        unique_metric,
        benchmark_and_optional_task_metric_scores_df,
    ) in enumerate(benchmark_and_optional_task_scores_df.groupby("metric")):
        if len(benchmark_and_optional_task_metric_scores_df) == 0:
            continue

        if (
            unique_metric not in pred_evals.globals.MULTIPLE_CHOICE_METRICS_SET
            and unique_metric not in pred_evals.globals.GENERATIVE_METRICS_SET
        ):
            continue

        plt.close()
        n_unique_models = benchmark_and_optional_task_metric_scores_df[
            "Model Nickname"
        ].nunique()

        avg_score_per_sample_idx = (
            benchmark_and_optional_task_metric_scores_df.groupby("sample_idx")
            .agg(
                {
                    "score": ["mean", "sem"],
                }
            )["score"]
            .sort_values("mean", ascending=False)
        )

        # Map from sample_idx to sorted_idx.
        sort_indices = avg_score_per_sample_idx.index.values
        # If sort_indices is [170, 193, 75, ....], then sort_idx_to_sample_idx is {170: 0, 193: 1, 75: 2, ...}.
        sample_idx_to_sort_idx = {
            sort_idx: sample_idx for sample_idx, sort_idx in enumerate(sort_indices)
        }
        benchmark_and_optional_task_metric_scores_df[
            "sort_idx"
        ] = benchmark_and_optional_task_metric_scores_df["sample_idx"].map(
            lambda sample_idx: sample_idx_to_sort_idx[sample_idx]
        )
        # Avoiding `ValueError: cannot reindex from a duplicate axis` mean?
        benchmark_and_optional_task_metric_scores_df.reset_index(
            drop=True, inplace=True
        )

        sns.lineplot(
            data=benchmark_and_optional_task_metric_scores_df,
            x="sort_idx",
            y="score",
            # linewidth=0,
            # marker="o",
            # markeredgecolor=None,
            errorbar=("se", 2),  # 95% CI, faster than "ci" for many samples
            hue=None,
        )

        plt.title(
            pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(
                benchmark_and_optional_task, benchmark_and_optional_task
            )
            + f" (Num. Models = {n_unique_models})"
        )
        plt.xlabel("Sample Index (Sorted)")
        plt.ylabel(
            pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT.get(
                unique_metric, unique_metric
            )
        )
        plt.ylim(
            pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT.get(
                unique_metric, (None, None)
            )
        )

        if unique_metric in {"prob_vocab_correct", "prob_choices_correct"}:
            if benchmark_and_optional_task_metric_scores_df["score"].min() > 0:
                plt.yscale("log")
            else:
                print(
                    f"Skipping log yscale for {benchmark_and_optional_task} {unique_metric} due to non-positive values."
                )

        save_plot_with_multiple_extensions(
            plot_dir=plot_dir,
            plot_title=f"{benchmark_and_optional_task}_{unique_metric}_{plot_title}",
        )

    # plt.show()


def plot_brier_score_vs_prob_vocab_correct_by_model_family(
    benchmark_and_optional_task_scores_wide_df: pd.DataFrame,
    benchmark_and_optional_task: str,
    plot_dir: str,
    plot_title: str = "brier_score_vs_prob_vocab_correct_by_model_family",
):
    # Plot prob_correct_vocab against prob_correct_choices.
    plt.close()
    plt.figure(figsize=(8, 4))
    g = sns.scatterplot(
        data=benchmark_and_optional_task_scores_wide_df,
        x="prob_vocab_correct",
        y="brier_score",
        hue="Model Family",
        s=10,
    )
    plt.xscale("log")
    plt.xlim(
        (
            np.nanquantile(
                benchmark_and_optional_task_scores_wide_df["prob_vocab_correct"], 0.01
            ),
            None,
        )
    )
    plt.ylim((0.0, None))
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.xlabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_vocab_correct"]
    )
    plt.ylabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["brier_score"],
    )
    plt.title(
        pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(
            benchmark_and_optional_task, benchmark_and_optional_task
        )
    )
    save_plot_with_multiple_extensions(
        plot_dir=plot_dir,
        plot_title=f"{benchmark_and_optional_task}_{plot_title}",
    )


def plot_coverage_frac_const_heatmap(
    all_model_all_benchmark_per_sample_results_df: pd.DataFrame,
    models_df: pd.DataFrame,
    plot_dir: str,
    plot_title: str = "model_evaluation_process_coverage_frac_const_heatmap",
):
    plt.close()
    coverage_frac_const_df = (
        all_model_all_benchmark_per_sample_results_df.groupby(
            ["benchmark_and_optional_task", "Model Nickname"]
        )
        .apply(lambda x: float(x["score"].nunique() == 1))
        .unstack()
    )
    # Reorder columns based on order of Model Nickname in models.csv.
    ordered_model_nicknames = [
        model_name
        for model_name in models_df["Model Nickname"]
        if model_name in coverage_frac_const_df.columns
    ]
    coverage_frac_const_df = coverage_frac_const_df[ordered_model_nicknames]
    plt.figure(figsize=(50, 20))
    g = sns.heatmap(
        coverage_frac_const_df,
        cmap="copper",
    )
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    save_plot_with_multiple_extensions(
        plot_dir=plot_dir,
        plot_title=plot_title,
    )
    # plt.show()


def plot_coverage_frac_nans_heatmap(
    all_model_all_benchmark_per_sample_results_df: pd.DataFrame,
    models_df: pd.DataFrame,
    plot_dir: str,
    plot_title: str = "model_evaluation_process_coverage_frac_nans_heatmap",
):
    plt.close()
    coverage_frac_nans_df = (
        all_model_all_benchmark_per_sample_results_df.groupby(
            ["benchmark_and_optional_task", "Model Nickname"]
        )
        .apply(lambda x: x["score"].isna().mean())
        .unstack()
    )
    # Reorder columns based on order of Model Nickname in models.csv.
    ordered_model_nicknames = [
        model_name
        for model_name in models_df["Model Nickname"]
        if model_name in coverage_frac_nans_df.columns
    ]
    coverage_frac_nans_df = coverage_frac_nans_df[ordered_model_nicknames]

    plt.figure(figsize=(50, 20))
    g = sns.heatmap(
        coverage_frac_nans_df,
        cmap="copper",
    )
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    save_plot_with_multiple_extensions(
        plot_dir=plot_dir,
        plot_title=plot_title,
    )
    # plt.show()


def plot_coverage_num_samples_heatmap(
    all_model_all_benchmark_per_sample_results_df: pd.DataFrame,
    models_df: pd.DataFrame,
    plot_dir: str,
    plot_title: str = "model_evaluation_process_coverage_num_samples_heatmap",
):
    plt.close()
    # What coverage do we have of models and benchmarks?
    # Plot as a heatmap.
    # Each column is a model, each row is a benchmark.
    coverage_num_samples_df = (
        all_model_all_benchmark_per_sample_results_df.groupby(
            ["benchmark_and_optional_task", "Model Nickname"]
        )
        .size()
        .unstack()
    )
    # Reorder columns based on order of Model Nickname in models.csv.
    ordered_model_nicknames = [
        model_name
        for model_name in models_df["Model Nickname"]
        if model_name in coverage_num_samples_df.columns
    ]
    coverage_num_samples_df = coverage_num_samples_df[ordered_model_nicknames]
    plt.figure(figsize=(50, 20))
    g = sns.heatmap(
        coverage_num_samples_df,
        cmap="Spectral_r",
        norm=matplotlib.colors.LogNorm(),
    )
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    save_plot_with_multiple_extensions(
        plot_dir=plot_dir,
        plot_title=plot_title,
    )
    # plt.show()


def plot_neg_log_prob_vocab_choices_vs_compute_by_sample_idx_col_model_family(
    benchmark_and_optional_task_scores_df: pd.DataFrame,
    benchmark_and_optional_task: str,
    plot_dir: str,
    plot_title: str = "neg_log_prob_vocab_choices_vs_compute_by_sample_idx_col_model_family",
    max_sample_idx_to_plot: int = 1000,
):
    if max_sample_idx_to_plot is not None:
        assert isinstance(max_sample_idx_to_plot, int)
        max_sample_idx_to_plot = min(
            max_sample_idx_to_plot,
            benchmark_and_optional_task_scores_df["sample_idx"].max(),
        )
        benchmark_and_optional_task_scores_df = benchmark_and_optional_task_scores_df[
            benchmark_and_optional_task_scores_df["sample_idx"] < max_sample_idx_to_plot
        ]

    min_compute, max_compute = (
        # models_df["Compute"].min(),
        1e18,  # Manually override smallest_compute because it is too small otherwise.
        1.1
        * benchmark_and_optional_task_scores_df[
            "Compute"
        ].max(),  # Add a little allowance for visualizing.
    )
    benchmark_and_optional_task_scores_df = benchmark_and_optional_task_scores_df[
        benchmark_and_optional_task_scores_df["Compute"] >= min_compute
    ]

    plt.close()
    # plt.figure(figsize=(8, 4))
    g = sns.relplot(
        data=benchmark_and_optional_task_scores_df,
        kind="line",
        x="Compute",
        y="neg_log_prob_vocab_choices",
        hue="sample_idx",
        col="Model Family",
        palette=sns.color_palette(["black"]),  # Make them all the same color (black).
        linewidth=0.25,
        legend=False,
        col_wrap=int(
            np.ceil(benchmark_and_optional_task_scores_df["Model Family"].nunique() / 2)
        ),
    )
    for ax in g.axes.flat:
        for line in ax.get_lines():
            line.set_alpha(0.05)
    g.set(
        xscale="log",
        xlim=(min_compute, max_compute),
        yscale="log",
    )
    g.set_axis_labels(
        x_var=pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["Compute"],
        y_var=pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[
            "neg_log_prob_vocab_choices"
        ],
    )
    g.set_titles(col_template="{col_name}")
    g.fig.suptitle(
        pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(
            benchmark_and_optional_task, benchmark_and_optional_task
        )
    )
    g.fig.subplots_adjust(top=0.9)
    save_plot_with_multiple_extensions(
        plot_dir=plot_dir,
        plot_title=f"{benchmark_and_optional_task}_{plot_title}",
    )
    # plt.show()


def plot_neg_log_prob_vocab_correct_vs_compute_by_sample_idx_col_model_family(
    benchmark_and_optional_task_scores_df: pd.DataFrame,
    benchmark_and_optional_task: str,
    plot_dir: str,
    plot_title: str = "neg_log_prob_vocab_correct_vs_compute_by_sample_idx_col_model_family",
    max_sample_idx_to_plot: int = 1000,
):
    if max_sample_idx_to_plot is not None:
        assert isinstance(max_sample_idx_to_plot, int)
        max_sample_idx_to_plot = min(
            max_sample_idx_to_plot,
            benchmark_and_optional_task_scores_df["sample_idx"].max(),
        )
        benchmark_and_optional_task_scores_df = benchmark_and_optional_task_scores_df[
            benchmark_and_optional_task_scores_df["sample_idx"] < max_sample_idx_to_plot
        ]

    min_compute, max_compute = (
        # models_df["Compute"].min(),
        1e18,  # Manually override smallest_compute because it is too small otherwise.
        1.1
        * benchmark_and_optional_task_scores_df[
            "Compute"
        ].max(),  # Add a little allowance for visualizing.
    )
    benchmark_and_optional_task_scores_df = benchmark_and_optional_task_scores_df[
        benchmark_and_optional_task_scores_df["Compute"] >= min_compute
    ]

    plt.close()
    g = sns.relplot(
        data=benchmark_and_optional_task_scores_df,
        kind="line",
        x="Compute",
        y="neg_log_prob_vocab_correct",
        hue="sample_idx",
        col="Model Family",
        palette=sns.color_palette(["black"]),  # Make them all the same color (black).
        linewidth=0.25,
        legend=False,
        col_wrap=int(
            np.ceil(benchmark_and_optional_task_scores_df["Model Family"].nunique() / 2)
        ),
    )
    for ax in g.axes.flat:
        for line in ax.get_lines():
            line.set_alpha(0.05)
    g.set(
        xscale="log",
        xlim=(min_compute, max_compute),
        yscale="log",
    )
    g.set_axis_labels(
        x_var=pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["Compute"],
        y_var=pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[
            "neg_log_prob_vocab_correct"
        ],
    )
    g.set_titles(col_template="{col_name}")
    g.fig.suptitle(
        pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(
            benchmark_and_optional_task, benchmark_and_optional_task
        )
    )
    g.fig.subplots_adjust(top=0.9)
    save_plot_with_multiple_extensions(
        plot_dir=plot_dir,
        plot_title=f"{benchmark_and_optional_task}_{plot_title}",
    )
    # plt.show()


def plot_prob_choices_correct_vs_prob_vocab_correct_and_acc_vs_prob_choices_correct_by_model_family_kde(
    benchmark_and_optional_task_scores_wide_df: pd.DataFrame,
    benchmark_and_optional_task: str,
    plot_dir: str,
    plot_title: str = "prob_choices_correct_vs_prob_vocab_correct_and_acc_vs_prob_choices_correct_by_model_family",
):
    # Plot prob_correct_vocab against prob_correct_choices.
    plt.close()
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 6),
        sharex=False,
        sharey=False,
    )
    g = sns.kdeplot(
        data=benchmark_and_optional_task_scores_wide_df,
        x="prob_vocab_correct",
        y="prob_choices_correct",
        hue="Model Family",
        ax=axes[0],
        legend=False,
    )
    axes[0].set_xlabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_vocab_correct"]
    )
    axes[0].set_ylabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_choices_correct"]
    )
    axes[0].set_xlim(
        pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT["prob_vocab_correct"],
    )
    axes[0].set_ylim(
        pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT["prob_choices_correct"],
    )
    # Cut off the extreme values to focus on bulk of distribution.
    axes[0].set_xlim(
        np.nanquantile(
            benchmark_and_optional_task_scores_wide_df["prob_vocab_correct"], 0.01
        ),
        1.05,
    )
    axes[0].set_xscale("log")

    # Define bins for 'p_correct_choices'
    bin_edges = np.linspace(0, 1, 31)  # 31 bins from 0 to 1
    benchmark_and_optional_task_scores_wide_df["bin"] = pd.cut(
        benchmark_and_optional_task_scores_wide_df["prob_choices_correct"],
        bins=bin_edges,
        include_lowest=True,
        # labels=np.round((bin_edges[:-1] + bin_edges[1:]) / 2, 2),
        labels=np.round(bin_edges[:-1], 2),
    )
    # Group by 'bin' and 'model_family', and calculate the fraction of accuracy = 1
    grouped = benchmark_and_optional_task_scores_wide_df.groupby(
        ["Model Family", "bin"]
    )
    fraction_of_accuracy_1 = (
        grouped["acc"].mean().reset_index()
    )  # 'mean' here represents the fraction of accuracy=1s
    l = sns.lineplot(
        data=fraction_of_accuracy_1,
        x="bin",
        y="acc",
        hue="Model Family",
        marker="o",
        ax=axes[1],
    )
    axes[1].axvline(x=0.5, color="black", linestyle="--")
    axes[1].set_xlabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_choices_correct"]
    )
    axes[1].set_ylabel("Fraction of Accuracy = 1")
    axes[1].set_xlim(0.0, 1.0)

    fig.suptitle(
        pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(
            benchmark_and_optional_task, benchmark_and_optional_task
        )
    )
    sns.move_legend(l, "upper left", bbox_to_anchor=(1, 1))
    save_plot_with_multiple_extensions(
        plot_dir=plot_dir,
        plot_title=f"{benchmark_and_optional_task}_{plot_title}_kde",
    )
    plt.close()


def plot_prob_choices_correct_vs_prob_vocab_correct_and_acc_vs_prob_choices_correct_by_model_family_scatter(
    benchmark_and_optional_task_scores_wide_df: pd.DataFrame,
    benchmark_and_optional_task: str,
    plot_dir: str,
    plot_title: str = "prob_choices_correct_vs_prob_vocab_correct_and_acc_vs_prob_choices_correct_by_model_family",
):
    # Plot prob_correct_vocab against prob_correct_choices.
    plt.close()
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 6),
        sharex=False,
        sharey=False,
    )
    g = sns.scatterplot(
        data=benchmark_and_optional_task_scores_wide_df,
        x="prob_vocab_correct",
        y="prob_choices_correct",
        hue="Model Family",
        s=5,
        # alpha=0.4,
        linewidth=0,
        ax=axes[0],
        legend=False,
    )
    axes[0].set_xlabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_vocab_correct"]
    )
    axes[0].set_ylabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_choices_correct"]
    )
    axes[0].set_xlim(
        pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT["prob_vocab_correct"],
    )
    axes[0].set_ylim(
        pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT["prob_choices_correct"],
    )
    # Cut off the extreme values to focus on bulk of distribution.
    axes[0].set_xlim(
        np.nanquantile(
            benchmark_and_optional_task_scores_wide_df["prob_vocab_correct"], 0.01
        ),
        1.05,
    )
    axes[0].set_xscale("log")

    # Define bins for 'p_correct_choices'
    bin_edges = np.linspace(0, 1, 31)  # 31 bins from 0 to 1
    benchmark_and_optional_task_scores_wide_df["bin"] = pd.cut(
        benchmark_and_optional_task_scores_wide_df["prob_choices_correct"],
        bins=bin_edges,
        include_lowest=True,
        # labels=np.round((bin_edges[:-1] + bin_edges[1:]) / 2, 2),
        labels=np.round(bin_edges[:-1], 2),
    )
    # Group by 'bin' and 'model_family', and calculate the fraction of accuracy = 1
    grouped = benchmark_and_optional_task_scores_wide_df.groupby(
        ["Model Family", "bin"]
    )
    fraction_of_accuracy_1 = (
        grouped["acc"].mean().reset_index()
    )  # 'mean' here represents the fraction of accuracy=1s
    l = sns.lineplot(
        data=fraction_of_accuracy_1,
        x="bin",
        y="acc",
        hue="Model Family",
        marker="o",
        ax=axes[1],
    )
    axes[1].axvline(x=0.5, color="black", linestyle="--")
    axes[1].set_xlabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_choices_correct"]
    )
    axes[1].set_ylabel("Fraction of Accuracy = 1")
    axes[1].set_xlim(0.0, 1.0)

    fig.suptitle(
        pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(
            benchmark_and_optional_task, benchmark_and_optional_task
        )
    )
    sns.move_legend(l, "upper left", bbox_to_anchor=(1, 1))
    save_plot_with_multiple_extensions(
        plot_dir=plot_dir,
        plot_title=f"{benchmark_and_optional_task}_{plot_title}_scatter",
    )
    plt.close()


def plot_prob_choices_correct_vs_prob_vocab_correct_and_acc_vs_prob_choices_correct_and_brier_score_vs_prob_choices_correct_by_model_family_scatter(
    benchmark_and_optional_task_scores_wide_df: pd.DataFrame,
    benchmark_and_optional_task: str,
    plot_dir: str,
    plot_title: str = "prob_choices_correct_vs_prob_vocab_correct_and_acc_vs_prob_choices_correct_and_brier_score_vs_prob_choices_correct_by_model_family_scatter",
):
    # Plot prob_correct_vocab against prob_correct_choices.
    plt.close()
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(20, 6),
        sharex=False,
        sharey=False,
    )
    g = sns.scatterplot(
        data=benchmark_and_optional_task_scores_wide_df,
        x="prob_vocab_correct",
        y="prob_choices_correct",
        hue="Model Family",
        s=5,
        # alpha=0.4,
        linewidth=0,
        ax=axes[0],
        legend=False,
    )
    axes[0].set_xlabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_vocab_correct"]
    )
    axes[0].set_ylabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_choices_correct"]
    )
    axes[0].set_xlim(
        pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT["prob_vocab_correct"],
    )
    axes[0].set_ylim(
        pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT["prob_choices_correct"],
    )
    # Cut off the extreme values to focus on bulk of distribution.
    axes[0].set_xlim(
        np.nanquantile(
            benchmark_and_optional_task_scores_wide_df["prob_vocab_correct"], 0.01
        ),
        1.05,
    )
    axes[0].set_xscale("log")

    # Define bins for 'p_correct_choices'
    bin_edges = np.linspace(0, 1, 31)  # 31 bins from 0 to 1
    benchmark_and_optional_task_scores_wide_df["bin"] = pd.cut(
        benchmark_and_optional_task_scores_wide_df["prob_choices_correct"],
        bins=bin_edges,
        include_lowest=True,
        # labels=np.round((bin_edges[:-1] + bin_edges[1:]) / 2, 2),
        labels=np.round(bin_edges[:-1], 2),
    )
    # Group by 'bin' and 'model_family', and calculate the fraction of accuracy = 1
    grouped = benchmark_and_optional_task_scores_wide_df.groupby(
        ["Model Family", "bin"]
    )
    fraction_of_accuracy_1 = (
        grouped["acc"].mean().reset_index()
    )  # 'mean' here represents the fraction of accuracy=1s
    sns.lineplot(
        data=fraction_of_accuracy_1,
        x="bin",
        y="acc",
        hue="Model Family",
        marker="o",
        ax=axes[1],
        legend=False,
    )
    axes[1].axvline(x=0.5, color="black", linestyle="--")
    axes[1].set_xlabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_choices_correct"]
    )
    axes[1].set_ylabel("Fraction of Accuracy = 1")
    axes[1].set_xlim(0.0, 1.0)

    g = sns.scatterplot(
        data=benchmark_and_optional_task_scores_wide_df,
        x="prob_choices_correct",
        y="brier_score",
        hue="Model Family",
        s=5,
        # alpha=0.4,
        linewidth=0,
        ax=axes[2],
    )
    axes[2].set_xlabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_choices_correct"]
    )
    axes[2].set_ylabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["brier_score"]
    )
    axes[2].set_xlim(
        pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT["prob_choices_correct"],
    )
    axes[2].set_ylim(
        pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT["brier_score"],
    )

    fig.suptitle(
        pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(
            benchmark_and_optional_task, benchmark_and_optional_task
        )
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    save_plot_with_multiple_extensions(
        plot_dir=plot_dir,
        plot_title=f"{benchmark_and_optional_task}_{plot_title}_scatter",
    )
    plt.close()


def plot_prob_choices_correct_vs_prob_vocab_correct_and_acc_vs_prob_vocab_correct_by_model_family_scatter(
    benchmark_and_optional_task_scores_wide_df: pd.DataFrame,
    benchmark_and_optional_task: str,
    plot_dir: str,
    plot_title: str = "prob_choices_correct_vs_prob_vocab_correct_and_acc_vs_prob_vocab_correct_by_model_family",
):
    # Plot prob_correct_vocab against prob_correct_choices.
    plt.close()
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(16, 6),
        sharex=False,
        sharey=False,
    )
    g = sns.scatterplot(
        data=benchmark_and_optional_task_scores_wide_df,
        x="prob_vocab_correct",
        y="prob_choices_correct",
        hue="Model Family",
        s=5,
        # alpha=0.4,
        linewidth=0,
        ax=axes[0],
        legend=False,
    )
    axes[0].set_xlabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_vocab_correct"]
    )
    axes[0].set_ylabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_choices_correct"]
    )
    axes[0].set_xlim(
        pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT["prob_vocab_correct"],
    )
    axes[0].set_ylim(
        pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT["prob_choices_correct"],
    )
    # Cut off the extreme values to focus on bulk of distribution.
    axes[0].set_xlim(
        np.nanquantile(
            benchmark_and_optional_task_scores_wide_df["prob_vocab_correct"], 0.01
        ),
        1.05,
    )
    axes[0].set_xscale("log")

    # Define bins for 'p_correct_choices'
    bin_edges = np.linspace(0, 1, 31)  # 31 bins from 0 to 1
    benchmark_and_optional_task_scores_wide_df["bin"] = pd.cut(
        benchmark_and_optional_task_scores_wide_df["prob_vocab_correct"],
        bins=bin_edges,
        include_lowest=True,
        # labels=np.round((bin_edges[:-1] + bin_edges[1:]) / 2, 2),
        labels=np.round(bin_edges[:-1], 2),
    )
    # Group by 'bin' and 'model_family', and calculate the fraction of accuracy = 1
    grouped = benchmark_and_optional_task_scores_wide_df.groupby(
        ["Model Family", "bin"]
    )
    fraction_of_accuracy_1 = (
        grouped["acc"].mean().reset_index()
    )  # 'mean' here represents the fraction of accuracy=1s
    l = sns.lineplot(
        data=fraction_of_accuracy_1,
        x="bin",
        y="acc",
        hue="Model Family",
        marker="o",
        ax=axes[1],
    )
    axes[1].axvline(x=0.5, color="black", linestyle="--")
    axes[1].set_xlabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_vocab_correct"]
    )
    axes[1].set_ylabel("Fraction of Accuracy = 1")
    axes[1].set_xlim(0.0, 1.0)

    fig.suptitle(
        pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(
            benchmark_and_optional_task, benchmark_and_optional_task
        )
    )
    sns.move_legend(l, "upper left", bbox_to_anchor=(1, 1))
    save_plot_with_multiple_extensions(
        plot_dir=plot_dir,
        plot_title=f"{benchmark_and_optional_task}_{plot_title}_scatter",
    )
    plt.close()


def plot_prob_choices_correct_vs_prob_vocab_correct_and_brier_score_vs_prob_vocab_correct_and_acc_vs_prob_vocab_correct_by_model_family_scatter(
    benchmark_and_optional_task_scores_wide_df: pd.DataFrame,
    benchmark_and_optional_task: str,
    plot_dir: str,
    plot_title: str = "prob_choices_correct_vs_prob_vocab_correct_and_brier_score_vs_prob_vocab_correct_and_acc_vs_prob_vocab_correct_by_model_family_scatter",
):
    # Plot prob_correct_vocab against prob_correct_choices.
    plt.close()
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(20, 6),
        sharex=False,
        sharey=False,
    )
    sns.scatterplot(
        data=benchmark_and_optional_task_scores_wide_df,
        x="prob_vocab_correct",
        y="prob_choices_correct",
        hue="Model Family",
        s=5,
        # alpha=0.4,
        linewidth=0,
        ax=axes[0],
        legend=False,
    )
    axes[0].set_xlabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_vocab_correct"]
    )
    axes[0].set_ylabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_choices_correct"]
    )
    axes[0].set_xlim(
        pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT["prob_vocab_correct"],
    )
    axes[0].set_ylim(
        pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT["prob_choices_correct"],
    )
    # Cut off the extreme values to focus on bulk of distribution.
    min_x_val = np.nanquantile(
        benchmark_and_optional_task_scores_wide_df["prob_vocab_correct"], 0.01
    )
    axes[0].set_xlim(min_x_val, 1.05)
    axes[0].set_xscale("log")

    sns.scatterplot(
        data=benchmark_and_optional_task_scores_wide_df,
        x="prob_vocab_correct",
        y="brier_score",
        hue="Model Family",
        s=5,
        # alpha=0.4,
        linewidth=0,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_xlabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_vocab_correct"]
    )
    axes[1].set_ylabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["brier_score"]
    )
    axes[1].set_xlim(min_x_val, 1.05)
    axes[1].set_xscale("log")
    axes[1].set_ylim(
        pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT["brier_score"],
    )

    # Define bins for 'p_vocab_choices'
    # bin_edges = np.quantile(
    #     benchmark_and_optional_task_scores_wide_df["prob_vocab_correct"],
    #     np.linspace(0.001, 1.0, 31),  # 31 bins from 0 to 1
    # )
    bin_edges = np.logspace(start=np.log10(min_x_val), stop=0.0, num=31)
    benchmark_and_optional_task_scores_wide_df["bin"] = pd.cut(
        benchmark_and_optional_task_scores_wide_df["prob_vocab_correct"],
        bins=bin_edges,
        include_lowest=True,
        labels=bin_edges[:-1],
    )
    # Group by 'bin' and 'model_family', and calculate the fraction of accuracy = 1
    grouped = benchmark_and_optional_task_scores_wide_df.groupby(
        ["Model Family", "bin"]
    )
    fraction_of_accuracy_1 = (
        grouped["acc"].mean().reset_index()
    )  # 'mean' here represents the fraction of accuracy=1s
    g = sns.lineplot(
        data=fraction_of_accuracy_1,
        x="bin",
        y="acc",
        hue="Model Family",
        marker="o",
        ax=axes[2],
    )
    axes[2].axvline(x=0.5, color="black", linestyle="--")
    axes[2].set_xlabel(
        pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT["prob_vocab_correct"]
    )
    axes[2].set_ylabel("Fraction of Accuracy = 1")
    axes[2].set_xlim(min_x_val, 1.05)
    axes[2].set_xscale("log")
    axes[2].set_ylim(
        pred_evals.globals.PERFORMANCE_METRICS_BOUNDS_DICT["acc"],
    )

    fig.suptitle(
        pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(
            benchmark_and_optional_task, benchmark_and_optional_task
        )
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    save_plot_with_multiple_extensions(
        plot_dir=plot_dir,
        plot_title=f"{benchmark_and_optional_task}_{plot_title}_scatter",
    )
    plt.close()


def plot_score_vs_compute_by_parameters_split_metric_split_model_family(
    benchmark_and_optional_task_scores_df: pd.DataFrame,
    models_df: pd.DataFrame,
    benchmark_and_optional_task: str,
    plot_dir: str,
    plot_title: str = "score_vs_compute_by_parameters",
):
    plt.close()

    min_compute, max_compute = (
        # models_df["Compute"].min(),
        1e13,  # Manually override smallest_compute because it is too small otherwise.
        1.1 * models_df["Compute"].max(),  # Add a little allowance for visualizing.
    )

    extended_model_family_benchmark_per_sample_results_df = (
        benchmark_and_optional_task_scores_df.merge(
            models_df,
            left_on="Model Nickname",
            right_on="Model Nickname",
            how="left",
        )
    )

    for (
        unique_metric,
        unique_model_family,
    ), metric_model_family_per_sample_scores in extended_model_family_benchmark_per_sample_results_df.groupby(
        [
            "metric",
            "Model Family",
        ]
    ):
        plt.close()

        if (
            unique_metric not in pred_evals.globals.MULTIPLE_CHOICE_METRICS_SET
            and unique_metric not in pred_evals.globals.GENERATIVE_METRICS_SET
        ):
            continue

        try:
            g = sns.lineplot(
                data=metric_model_family_per_sample_scores,
                x="Compute",
                y="score",
                hue="Parameters",
                errorbar=("ci", 95),
                err_style="bars",
                hue_norm=matplotlib.colors.LogNorm(),
            )
            g = sns.scatterplot(
                data=metric_model_family_per_sample_scores.groupby(
                    ["Compute", "Parameters"]
                )
                .agg({"score": "mean"})
                .reset_index(),
                x="Compute",
                y="score",
                hue="Parameters",
                hue_norm=matplotlib.colors.LogNorm(),
                legend=False,
                ax=g,
            )
        except Exception as e:
            print(
                "Error at ",
                benchmark_and_optional_task,
                unique_metric,
                unique_model_family,
            )

        # Set seaborn legend.
        plt.xlabel(r"Compute (6ND)")
        if unique_metric in {
            "prob_vocab_correct",
        }:
            plt.yscale("log")
        elif unique_metric in {
            "log_prob_vocab_correct",
        }:
            # This is a hack; we really want a negative log.
            plt.yscale("symlog")

        plt.ylabel(
            pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT.get(
                unique_metric, unique_metric
            )
        )
        plt.xlim(min_compute, max_compute)
        plt.xscale("log")

        # https://stackoverflow.com/a/54213918/4570472
        plt.title(
            pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(
                benchmark_and_optional_task, benchmark_and_optional_task
            )
        )
        save_plot_with_multiple_extensions(
            plot_dir=plot_dir,
            plot_title=f"{benchmark_and_optional_task}_{unique_metric}_{unique_model_family}_{plot_title}",
        )
        # plt.show()


def plot_target_is_greedy_vs_prob_choices_correct_by_model_family(
    benchmark_and_optional_task_scores_wide_df: pd.DataFrame,
    benchmark_and_optional_task: str,
    plot_dir: str,
    plot_title: str = "target_is_greedy_vs_prob_choices_correct_by_model_family",
):
    # Plot fraction of target_is_greedy against prob_correct_choices.
    plt.close()
    plt.figure(figsize=(8, 4))

    # Define bins for 'p_correct_choices'
    bin_edges = np.linspace(0, 1, 31)  # 31 bins from 0 to 1
    benchmark_and_optional_task_scores_wide_df["bin"] = pd.cut(
        benchmark_and_optional_task_scores_wide_df["prob_choices_correct"],
        bins=bin_edges,
        include_lowest=True,
        # labels=np.round((bin_edges[:-1] + bin_edges[1:]) / 2, 2),
        labels=np.round(bin_edges[:-1], 2),
    )
    # Group by 'bin' and 'model_family', and calculate the fraction of accuracy = 1
    grouped = benchmark_and_optional_task_scores_wide_df.groupby(
        ["Model Family", "bin"]
    )
    fraction_of_target_is_greedy_df = (
        grouped["target_is_greedy"].mean().reset_index()
    )  # 'mean' here represents the fraction of accuracy=1s
    g = sns.lineplot(
        data=fraction_of_target_is_greedy_df,
        x="bin",
        y="target_is_greedy",
        hue="Model Family",
        marker="o",
    )
    g.set(
        xlabel=pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[
            "prob_choices_correct"
        ],
        xlim=(0.0, 1.0),
        ylabel="Fraction of Target-Is-Greedy = 1",
        ylim=(0.0, 1.0),
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    plt.title(
        pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(
            benchmark_and_optional_task, benchmark_and_optional_task
        )
    )
    save_plot_with_multiple_extensions(
        plot_dir=plot_dir,
        plot_title=f"{benchmark_and_optional_task}_{plot_title}",
    )
    plt.close()


def plot_target_is_greedy_vs_prob_vocab_correct_by_model_family(
    benchmark_and_optional_task_scores_wide_df: pd.DataFrame,
    benchmark_and_optional_task: str,
    plot_dir: str,
    plot_title: str = "target_is_greedy_vs_prob_vocab_correct_by_model_family",
):
    # Plot fraction of target_is_greedy against prob_correct_choices.
    plt.close()
    plt.figure(figsize=(8, 4))

    # Define bins for 'p_vocab_correct'
    bin_edges = np.nanquantile(
        a=benchmark_and_optional_task_scores_wide_df["prob_vocab_correct"],
        q=np.linspace(0.01, 1.0, 31),
    )

    benchmark_and_optional_task_scores_wide_df["bin"] = pd.cut(
        benchmark_and_optional_task_scores_wide_df["prob_vocab_correct"],
        bins=bin_edges,
        include_lowest=True,
        labels=bin_edges[:-1],
    )
    # Group by 'bin' and 'model_family', and calculate the fraction of accuracy = 1
    grouped = benchmark_and_optional_task_scores_wide_df.groupby(
        ["Model Family", "bin"]
    )
    fraction_of_target_is_greedy_df = (
        grouped["target_is_greedy"].mean().reset_index()
    )  # 'mean' here represents the fraction of accuracy=1s
    g = sns.lineplot(
        data=fraction_of_target_is_greedy_df,
        x="bin",
        y="target_is_greedy",
        hue="Model Family",
        marker="o",
    )
    g.set(
        xlabel=pred_evals.globals.PERFORMANCE_METRICS_NICE_STRINGS_DICT[
            "prob_vocab_correct"
        ],
        xscale="log",
        ylabel="Fraction of Target-Is-Greedy = 1",
        ylim=(0.0, 1.0),
    )
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    # If we try to do this with g.set(), the x-axis gets compressed to the far right
    # and cannot be read. I do not know why.
    plt.xlim(None, 1.0)
    plt.title(
        pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(
            benchmark_and_optional_task, benchmark_and_optional_task
        )
    )
    save_plot_with_multiple_extensions(
        plot_dir=plot_dir,
        plot_title=f"{benchmark_and_optional_task}_{plot_title}",
    )
    # plt.show()
    # plt.close()


def save_plot_with_multiple_extensions(plot_dir: str, plot_title: str):
    # Ensure that axis labels don't overlap.
    plt.gcf().tight_layout()

    extensions = [
        "pdf",
        "png",
    ]
    for extension in extensions:
        plot_path = os.path.join(plot_dir, plot_title + f".{extension}")
        print(f"Plotted {plot_path}")
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
