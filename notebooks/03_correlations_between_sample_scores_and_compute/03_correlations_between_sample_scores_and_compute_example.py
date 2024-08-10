import itertools
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from pred_evals.globals import PERFORMANCE_METRICS_NICE_STRINGS_DICT
from pred_evals.metrics import ALL_METRICS_BOUNDS_DICT
import pred_evals.plot
import pred_evals.utils


data_dir, results_dir = pred_evals.utils.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

results_dir = os.path.join(results_dir, "examples")
os.makedirs(results_dir, exist_ok=True)


num_samples = int(1e6)
perfectly_correlated = np.full(shape=num_samples, fill_value=0.99)
perfectly_anticorrelated = np.full(shape=num_samples, fill_value=-0.99)
perfectly_uncorrelated = np.random.normal(loc=0, scale=0.01, size=num_samples)
perfectly_uniform = np.random.uniform(low=-1.0, high=1.0, size=num_samples)
# Hijack to trick seaborn into plotting horizontal lines.
perfectly_correlated[:20] = -1.0
perfectly_anticorrelated[-20:] = 1.0
perfectly_uncorrelated[:10] = -1.0
perfectly_uncorrelated[-10:] = 1.0


df = pd.DataFrame.from_dict(
    {
        "Correlation between FLOPS and Scores (Per Sample)": np.concatenate(
            [
                perfectly_correlated,
                perfectly_anticorrelated,
                perfectly_uncorrelated,
                perfectly_uniform,
            ]
        ),
        "Example Distribution": np.concatenate(
            [
                np.array(["Correlated" for _ in range(num_samples)]),
                np.array(["Anticorrelated" for _ in range(num_samples)]),
                np.array(["Uncorrelated" for _ in range(num_samples)]),
                np.array(["Uniform" for _ in range(num_samples)]),
            ]
        ),
    }
)

plt.close()
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(14, 5))
sns.histplot(
    data=df,
    x="Correlation between FLOPS and Scores (Per Sample)",
    hue="Example Distribution",
    stat="probability",
    common_norm=False,
    ax=axes[0],
    legend=False,
    alpha=0.9,
)
axes[0].set_xlim(-1.0, 1.0)
axes[0].set_ylim(1e-2, 1.0)
axes[0].set_yscale("log")
axes[0].set_xlabel("Correlation between FLOPS and Scores (Per Sample)")
axes[0].set_ylabel("Fraction of Samples")
g = sns.ecdfplot(
    data=df,
    x="Correlation between FLOPS and Scores (Per Sample)",
    hue="Example Distribution",
    # common_norm=False,
    complementary=True,
    ax=axes[1],
    linewidth=5,
)
axes[1].set_xlim(-1.0, 1.0)
axes[1].set_ylim(-0.01, 1.01)
axes[1].set_ylabel("1 - ECDF")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
fig.suptitle("Example Distributions of Score-Compute Correlations")
pred_evals.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir,
    plot_title="example_correlation_btwn_flops_and_scores_by_distribution",
)
# plt.show()

print("Finished notebooks/03_correlations_between_sample_scores_and_compute_example!")
