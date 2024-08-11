from functools import partial
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union
import wandb


def add_additional_compute_columns(df: pd.DataFrame) -> pd.DataFrame:
    assert "Compute" in df
    df["Compute / Max Compute"] = df["Compute"] / df["Compute"].max()
    df["Log10 Compute"] = np.log10(df["Compute"])
    df["LogE Compute"] = np.log(df["Compute"])
    df["Log10 (Compute / Max Compute)"] = np.log10(df["Compute"] / df["Compute"].max())
    df["LogE (Compute / Max Compute)"] = np.log(df["Compute"] / df["Compute"].max())
    return df


def add_log_compute_column(df: pd.DataFrame) -> pd.DataFrame:
    assert "Compute" in df
    df["Log10 Compute"] = np.log10(df["Compute"])
    df["LogE Compute"] = np.log(df["Compute"])
    return df


def add_scaled_compute_column(df: pd.DataFrame) -> pd.DataFrame:
    assert "Compute" in df
    df["Compute / Max Compute"] = df["Compute"] / df["Compute"].max()
    return df


def add_log_scaled_compute_column(df: pd.DataFrame) -> pd.DataFrame:
    assert "Compute" in df
    df["Log10 (Compute / Max Compute)"] = np.log10(df["Compute"] / df["Compute"].max())
    df["LogE (Compute / Max Compute)"] = np.log(df["Compute"] / df["Compute"].max())
    return df


def calculate_brier_score_multiple_choice(
    log_likelihoods: np.ndarray,
    target_idx: int,
    epsilon: float = 1e-9,
) -> float:
    """
    Brier score: the mean squared error between the probabilities and the one-hot targets.
    treat each answer choice as a one-hot target and the probabilities as a prediction.
    Note: For multiple-choice benchmarks only.
    """
    softmax_probs = softmax(log_likelihoods, epsilon=epsilon)
    one_hot_target = np.zeros(len(softmax_probs))
    one_hot_target[target_idx] = 1
    brier_score = np.mean(np.square(softmax_probs - one_hot_target))
    return brier_score


def download_wandb_project_runs_configs(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str] = None,
    finished_only: bool = True,
    refresh: bool = False,
    wandb_username: str = None,
) -> pd.DataFrame:
    runs_configs_df_path = os.path.join(
        data_dir, "sweeps=" + ",".join(sweep_ids) + "_runs_configs.csv"
    )
    if refresh or not os.path.isfile(runs_configs_df_path):
        # Download sweep results
        api = wandb.Api(timeout=600)

        if wandb_username is None:
            wandb_username = api.viewer.username

        sweep_results_list = []
        for sweep_id in sweep_ids:
            sweep = api.sweep(f"{wandb_username}/{wandb_project_path}/{sweep_id}")
            for run in tqdm(sweep.runs):
                # .summary contains the output keys/values for metrics like accuracy.
                #  We call ._json_dict to omit large files
                summary = run.summary._json_dict

                # .config contains the hyperparameters.
                #  We remove special values that start with _.
                summary.update(
                    {k: v for k, v in run.config.items() if not k.startswith("_")}
                )

                summary.update(
                    {
                        "State": run.state,
                        "Sweep": run.sweep.id if run.sweep is not None else None,
                        "run_id": run.id,
                    }
                )
                # .name is the human-readable name of the run.
                summary.update({"run_name": run.name})
                sweep_results_list.append(summary)

            runs_configs_df = pd.DataFrame(sweep_results_list)

            # Save to disk.
            runs_configs_df.to_csv(runs_configs_df_path, index=False)
            print(f"Wrote {runs_configs_df_path} to disk.")
    else:
        runs_configs_df = pd.read_csv(runs_configs_df_path)
        print(f"Loaded {runs_configs_df_path} from disk.")

    # Keep only finished runs
    finished_runs = runs_configs_df["State"] == "finished"
    print(
        f"% of successfully finished runs: {100.0 * finished_runs.mean()} ({finished_runs.sum()} / {len(finished_runs)})"
    )

    if finished_only:
        runs_configs_df = runs_configs_df[finished_runs]

        # Check that we don't have an empty data frame.
        assert len(runs_configs_df) > 0

        # Ensure we aren't working with a slice.
        runs_configs_df = runs_configs_df.copy()

    return runs_configs_df


def download_wandb_project_runs_histories(
    wandb_project_path: str,
    data_dir: str,
    sweep_ids: List[str] = None,
    wandb_run_history_samples: int = 10000,
    refresh: bool = False,
    keys: List[str] = None,
    wandb_username: str = None,
    filetype: str = "csv",
) -> pd.DataFrame:
    if keys is None:
        keys = ["losses_train/loss_epoch", "losses_val/loss"]

    assert filetype in {"csv", "parquet", "feather"}

    runs_histories_df_path = os.path.join(
        data_dir, "sweeps=" + ",".join(sweep_ids) + f"_runs_histories.{filetype}"
    )
    if refresh or not os.path.isfile(runs_histories_df_path):
        # Download sweep results
        api = wandb.Api(timeout=6000)

        if wandb_username is None:
            wandb_username = api.viewer.username

        runs_histories_list = []
        for sweep_id in sweep_ids:
            sweep = api.sweep(f"{wandb_username}/{wandb_project_path}/{sweep_id}")
            for run in tqdm(sweep.runs):
                # https://community.wandb.ai/t/run-history-returns-different-values-on-almost-each-call/2431/4
                history = run.history(
                    samples=wandb_run_history_samples,
                )
                if history.empty:
                    continue
                history["run_id"] = run.id
                runs_histories_list.append(history)

        assert len(runs_histories_list) > 0
        runs_histories_df = pd.concat(runs_histories_list, sort=False)

        runs_histories_df.sort_values(["run_id"], ascending=True, inplace=True)

        runs_histories_df.reset_index(inplace=True, drop=True)

        if filetype == "csv":
            runs_histories_df.to_csv(runs_histories_df_path, index=False)
        elif filetype == "feather":
            runs_histories_df.reset_index(inplace=True)
            runs_histories_df.to_feather(runs_histories_df_path)
        elif filetype == "parquet":
            runs_histories_df.to_parquet(runs_histories_df_path, index=False)
        else:
            raise ValueError(f"Invalid filetype: {filetype}")
        print(f"Wrote {runs_histories_df_path} to disk")
    else:
        if filetype == "csv":
            runs_histories_df = pd.read_csv(runs_histories_df_path)
        elif filetype == "feather":
            runs_histories_df = pd.read_feather(runs_histories_df_path)
        elif filetype == "parquet":
            runs_histories_df = pd.read_parquet(runs_histories_df_path)
            runs_histories_df = pd.read_parquet(runs_histories_df_path)
        else:
            raise ValueError(f"Invalid filetype: {filetype}")
        print(f"Loaded {runs_histories_df_path} from disk.")

    return runs_histories_df


def extract_per_metric_per_sample_results_from_df(
    model_benchmark_per_sample_results_df: pd.DataFrame,
    benchmark_and_optional_task: str,
) -> pd.DataFrame:
    # Create a "wide" pandas dataframe where each column represents a different metric.
    metrics_to_scores_dict = {
        "sample_idx": model_benchmark_per_sample_results_df["doc_id"].values,
    }
    # Extract standard metrics if they exist.
    if "acc" in model_benchmark_per_sample_results_df.columns:
        metrics_to_scores_dict["acc"] = model_benchmark_per_sample_results_df[
            "acc"
        ].values
    if "acc_norm" in model_benchmark_per_sample_results_df.columns:
        metrics_to_scores_dict["acc_norm"] = model_benchmark_per_sample_results_df[
            "acc_norm"
        ].values
    if "em" in model_benchmark_per_sample_results_df.columns:
        metrics_to_scores_dict["em"] = model_benchmark_per_sample_results_df[
            "em"
        ].values
        metrics_to_scores_dict["log_em"] = np.log(metrics_to_scores_dict["em"])
    if "exact_match" in model_benchmark_per_sample_results_df.columns:
        metrics_to_scores_dict["em"] = model_benchmark_per_sample_results_df[
            "exact_match"
        ].values
        metrics_to_scores_dict["log_em"] = np.log(metrics_to_scores_dict["em"])
    if "f1" in model_benchmark_per_sample_results_df.columns:
        if model_benchmark_per_sample_results_df["f1"].dtype == np.float64:
            metrics_to_scores_dict["f1"] = model_benchmark_per_sample_results_df[
                "f1"
            ].values
        else:
            pass
            # TODO: Debug why mctaco has f1 as a list of of 2 integers.
    if "f1_abstractive" in model_benchmark_per_sample_results_df.columns:
        metrics_to_scores_dict[
            "f1_abstractive"
        ] = model_benchmark_per_sample_results_df["f1_abstractive"].values

    if "perplexity" in model_benchmark_per_sample_results_df.columns:
        metrics_to_scores_dict["perplexity"] = model_benchmark_per_sample_results_df[
            "perplexity"
        ].values

    # Extra more complicated metrics based on log likelihood(s).
    try:
        additional_metrics_df: pd.DataFrame = (
            model_benchmark_per_sample_results_df.apply(
                partial(
                    extract_per_sample_benchmark_specific_metrics_from_df,
                    benchmark_and_optional_task=benchmark_and_optional_task,
                ),
                axis=1,
            )
        )

        # Transfer additional metrics over to the metrics_to_scores_dict.
        for metric in additional_metrics_df.columns:
            metrics_to_scores_dict[metric] = additional_metrics_df[metric].values

    except AssertionError as e:
        print(e)

    # Compute perplexity as natural exponent of log likelihood.
    if "log_prob_vocab_correct" in metrics_to_scores_dict:
        metrics_to_scores_dict["neg_log_prob_vocab_correct"] = -metrics_to_scores_dict[
            "log_prob_vocab_correct"
        ]
    prob_keys = [
        "prob_vocab_choices",
        "prob_vocab_correct",
        "prob_vocab_incorrect",
        "prob_choices_correct",
    ]
    for prob_key in prob_keys:
        if prob_key in metrics_to_scores_dict:
            # e.g., prob_vocab_choices -> ppl_vocab_choices
            metrics_to_scores_dict[prob_key.replace("prob_", "ppl_")] = (
                1.0 / metrics_to_scores_dict[prob_key]
            )
            # e.g., prob_vocab_choices -> log_prob_vocab_choices
            metrics_to_scores_dict[f"log_{prob_key}"] = np.log(
                metrics_to_scores_dict[prob_key]
            )
            # e.g., log_prob_vocab_choices -> neg_log_prob_vocab_choices
            metrics_to_scores_dict[f"neg_log_{prob_key}"] = -metrics_to_scores_dict[
                f"log_{prob_key}"
            ]

    keys_and_bounds = {
        "acc": (0.0, 1.0),
        "acc_norm": (0.0, 1.0),
        "em": (0.0, 1.0),
        "f1": (0.0, 1.0),
        "f1_abstractive": (0.0, 1.0),
        # "perplexity": (0.0, np.inf),  # For some reason, Eleuther AI's perplexity is incorrect.
        "prob_vocab_correct": (0.0, 1.0),
        "prob_choices_correct": (0.0, 1.0),
    }
    for key, bounds in keys_and_bounds.items():
        if key in metrics_to_scores_dict:
            values = metrics_to_scores_dict[key]
            # NaNs throw the assert statements unnecessarily.
            values = values[~np.isnan(values)]
            assert np.all(bounds[0] <= values)
            assert np.all(values <= bounds[1])

    wide_model_benchmark_per_sample_results_df = pd.DataFrame.from_dict(
        metrics_to_scores_dict
    )

    tall_model_benchmark_per_sample_results_df = (
        wide_model_benchmark_per_sample_results_df.melt(
            id_vars="sample_idx", var_name="metric", value_name="score"
        )
    )

    return tall_model_benchmark_per_sample_results_df


def extract_per_sample_benchmark_specific_metrics_from_df(
    row: pd.Series,
    benchmark_and_optional_task: str,
) -> pd.Series:
    # Information from Hailey Schoelkopf:
    # filtered_resps will be of type `List[tuple[float, bool]]` where the length = the total number
    # of possible target strings / answer choices (in lambada's case, one target string).
    # and each entry is [<floating point number, being loglikelihood = sum of loglikelihoods of each
    # token prediction in the target string>, <True/False where is True iff the target string would be
    # the prediction output by greedy generation for len(target string) tokens>].

    # For the metric keys: for perplexity this is a bit weird, sorry. For wikitext, the entries are
    # [<loglikelihood of whole string>, <length in words / bytes of whole string>] and these get ingested
    # using https://github.com/EleutherAI/lm-evaluation-harness/blob/19cbb2923a9fe0549754b8b5cb7511c9832a5851/lm_eval/api/metrics.py#L42
    # to compute perplexity at the dataset level, weighting by each example's size

    additional_metrics_dict = {}

    if (
        benchmark_and_optional_task
        in {
            "arc_easy",
            "arc_challenge",
            "hellaswag",
            "logiqa",
            "logiqa2",
            "mathqa",
            "mc_taco",
            "openbookqa",
            "piqa",
            "prost",
            "pubmedqa",
            "qasper_bool",
            "race",
            "sciq",
            "social_iqa",
            "swag",
            "winogrande",
            "xwinograd_en",
        }
        or benchmark_and_optional_task.startswith("anli")
        or benchmark_and_optional_task.startswith("advanced_ai_risk")
        or benchmark_and_optional_task.startswith("ethics")
        or benchmark_and_optional_task.startswith("mmlu")
        or benchmark_and_optional_task.startswith("persona")
        or benchmark_and_optional_task.startswith("sycophancy")
    ):
        target_idx: Union[int, str] = row["target"]
        if benchmark_and_optional_task.startswith("anli"):
            # "anli_r1", "_r2" and "_r3" store the target as strings "True", "False" or "Neither".
            target_idx: List[int] = [
                idx
                for idx, arg in enumerate(row["arguments"])
                if arg[1].endswith(target_idx)
            ]
            assert len(target_idx) == 1
            target_idx = target_idx[0]
        elif benchmark_and_optional_task == "pubmedqa":
            # pubmedqa stores the target as "yes", "no" or "maybe". We need to find the correct index.
            # This information is stored in the arguments but with a leading space, e.g., " yes".
            target_str = f" {target_idx}"
            target_idx: List[int] = [
                idx for idx, arg in enumerate(row["arguments"]) if arg[1] == target_str
            ]
            assert len(target_idx) == 1
            target_idx = target_idx[0]
        elif benchmark_and_optional_task in {"winogrande", "xwinograd_en"}:
            # TODO: Debug why winogrande doesn't have the correct target.
            target_idx = (
                int(row["doc"]["answer"]) - 1
            )  # These datasets use 1-based indexing.

        # The first elements are the log likelihoods.
        log_likelihoods = np.array([ele[0] for ele in row["filtered_resps"]])
        additional_metrics_dict["num_choices"] = len(log_likelihoods)
        additional_metrics_dict["log_prob_vocab_correct"] = log_likelihoods[target_idx]

        prob_vocab_choices = np.exp(log_likelihoods)
        additional_metrics_dict["prob_vocab_correct"] = prob_vocab_choices[target_idx]
        additional_metrics_dict["prob_vocab_choices"] = np.sum(prob_vocab_choices)
        additional_metrics_dict["prob_vocab_incorrect"] = (
            additional_metrics_dict["prob_vocab_choices"]
            - additional_metrics_dict["prob_vocab_correct"]
        )

        prob_choices = softmax(log_likelihoods)
        additional_metrics_dict["prob_choices_correct"] = prob_choices[target_idx]
        additional_metrics_dict["brier_score"] = calculate_brier_score_multiple_choice(
            log_likelihoods=log_likelihoods, target_idx=target_idx
        )

        for idx, prob_choice_vocab in enumerate(prob_vocab_choices):
            additional_metrics_dict[f"prob_vocab_choice={idx}"] = prob_choice_vocab
            additional_metrics_dict[
                f"log_likelihood_vocab_choice={idx}"
            ] = log_likelihoods[idx]
            additional_metrics_dict[
                f"neg_log_likelihood_vocab_choice={idx}"
            ] = -log_likelihoods[idx]
            additional_metrics_dict[f"prob_choices_choice={idx}"] = prob_choices[idx]
            additional_metrics_dict[f"log_likelihood_choices_choice={idx}"] = np.log(
                prob_choices[idx]
            )
            additional_metrics_dict[
                f"neg_log_likelihood_choices_choice={idx}"
            ] = -additional_metrics_dict[f"log_likelihood_choices_choice={idx}"]

        # calculate exact-match accuracy
        # (is target string greedily generated, globally)
        # -- different from 'acc' (is target string the most probable among choices
        target_is_greedy = float(row["filtered_resps"][target_idx][1])
        additional_metrics_dict["target_is_greedy"] = target_is_greedy
    elif benchmark_and_optional_task in {"squadv2"}:
        # TODO: Grab the other available metrics.
        additional_metrics_dict["log_prob_vocab_correct"] = row["filtered_resps"][1][0]
    # These tasks check the probability mass the model places on the correct answer.
    elif benchmark_and_optional_task in {"asdiv", "lambada_openai", "lambada_standard"}:
        # row["target"] is the string of the correct answer. There's only one filtered response.
        additional_metrics_dict["log_prob_vocab_correct"] = row["filtered_resps"][0][0]
        additional_metrics_dict["prob_vocab_correct"] = np.exp(
            additional_metrics_dict["log_prob_vocab_correct"]
        )
    elif benchmark_and_optional_task in {"qasper_freeform"}:
        # Unknown benchmark_and_optional_task: qasper_freeform  # f1 abstractive
        # TODO: Figure out what we want from this.
        additional_metrics_dict["log_prob_vocab_correct"] = np.nan
    # These tasks use Exact Match and thus have no negative log likelihood.
    elif (
        benchmark_and_optional_task
        in {
            "babi",
            "drop",
            "fld",
            "nq_open",
            "triviaqa",
            "unscramble",
        }
        or benchmark_and_optional_task.startswith("bbh")
        or benchmark_and_optional_task.startswith("gsm8k")
        or benchmark_and_optional_task.startswith("minerva")
    ):
        additional_metrics_dict["log_prob_vocab_correct"] = np.nan
    # These tasks use perplexity instead of negative log likelihood.
    elif benchmark_and_optional_task in {"wikitext"}:
        # TODO: debug why their perplexities are negative. This is not valid.
        # log_likelihood = np.log(row["filtered_resps"][0])
        # target_log_likelihood = np.nan
        additional_metrics_dict["unknown"] = row["filtered_resps"][0]
    elif benchmark_and_optional_task in {"webqs"}:
        # TODO: Decide what to do with multiple targets.
        # Multiple targets are possible because there are multiple valid answers.
        # e.g. What language do Jamaican people speak? Answers: Jamaican English, Jamaican Creole English
        additional_metrics_dict["log_prob_vocab_correct"] = np.nan
    else:
        raise ValueError(
            f"Unknown benchmark_and_optional_task: {benchmark_and_optional_task}"
        )

    additional_metrics_series = pd.Series(additional_metrics_dict)
    return additional_metrics_series


def load_models_df(models_df_path: str = "configs/models.csv") -> pd.DataFrame:
    models_df = pd.read_csv(models_df_path, index_col=None)
    models_df["Compute"] = 6.0 * models_df["Parameters"] * models_df["Tokens"]
    return models_df


def retrieve_wandb_username() -> str:
    import wandb

    api = wandb.Api(timeout=30)
    wandb_username = api.viewer.username
    return wandb_username


def setup_notebook_dir(
    notebook_dir: str,
    refresh: bool = False,
) -> Tuple[str, str]:
    # Declare paths.
    data_dir = os.path.join(notebook_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    results_dir = os.path.join(notebook_dir, "results")
    if refresh:
        import shutil

        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    return data_dir, results_dir


def softmax(x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / (epsilon + np.sum(e_x))
