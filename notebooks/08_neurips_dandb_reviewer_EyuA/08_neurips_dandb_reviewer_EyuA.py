import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import pred_evals.globals
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
all_model_all_benchmark_per_sample_results_df = pd.read_feather(
    f"eval_results/all_model_benchmark_per_sample_results_{date_str}.feather"
)

# Take only arc_challenge, arc_easy, hellaswag, with metric = accuracy
all_model_all_benchmark_per_sample_results_df = (
    all_model_all_benchmark_per_sample_results_df[
        (
            all_model_all_benchmark_per_sample_results_df[
                "benchmark_and_optional_task"
            ].isin(["arc_challenge", "arc_easy", "hellaswag"])
        )
        & (all_model_all_benchmark_per_sample_results_df["metric"] == "acc")
    ]
)

# Compute average score per model per benchmark.
all_model_all_benchmark_avg_acc_df = (
    all_model_all_benchmark_per_sample_results_df.groupby(
        ["benchmark_and_optional_task", "Model Nickname"]
    )["score"]
    .mean()
    .reset_index()
)

# Make benchmarks and optional tasks more readable.
all_model_all_benchmark_avg_acc_df[
    "benchmark_and_optional_task"
] = all_model_all_benchmark_avg_acc_df["benchmark_and_optional_task"].map(
    lambda k: pred_evals.globals.BENCHMARKS_NICE_STRINGS_DICT.get(k, k)
)


# Merge to add Model Family and Compute.
extended_all_model_all_benchmark_avg_acc_df = all_model_all_benchmark_avg_acc_df.merge(
    models_df,
    left_on="Model Nickname",
    right_on="Model Nickname",
    how="left",
)

# Exclude the one starting checkpoint with 0 FLOP because it screws up the log scaling.
extended_all_model_all_benchmark_avg_acc_df = (
    extended_all_model_all_benchmark_avg_acc_df[
        extended_all_model_all_benchmark_avg_acc_df["Compute"] > 0
    ]
)

# Exclude Llama 2 family because it wasn't ready in time for NeurIPS D&B submission.
extended_all_model_all_benchmark_avg_acc_df = (
    extended_all_model_all_benchmark_avg_acc_df[
        extended_all_model_all_benchmark_avg_acc_df["Model Family"] != "Llama 2"
    ]
)

# Plot scaling curves.
plt.close()
min_compute, max_compute = (
    # models_df["Compute"].min(),
    1e13,  # Manually override smallest_compute because it is too small otherwise.
    1.1 * models_df["Compute"].max(),  # Add a little allowance for visualizing.
)
g = sns.relplot(
    data=extended_all_model_all_benchmark_avg_acc_df,
    kind="line",
    x="Compute",
    y="score",
    hue="Model Family",
    col="benchmark_and_optional_task",
    # row="Model Family",
    facet_kws={"sharex": "row", "sharey": True, "margin_titles": True},
)
g.set(xscale="log", ylabel="Accuracy (Average)", ylim=(0, 1))
g.set_titles(col_template="{col_name}", row_template="{row_name}")
# Add dashed horizontal line at 0.25 accuracy.
for ax in g.axes.flat:
    ax.axhline(0.25, ls="--", color="black")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
pred_evals.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title="reviewer_EyuA_accuracy_scaling_curves",
)
plt.show()

print("Finished notebooks/08_neurips_dandb_reviewer_EyuA!")
