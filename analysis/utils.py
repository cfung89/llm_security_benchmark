import pandas as pd
import numpy as np
from typing import Callable
from inspect_ai.log import EvalLog, EvalSampleSummary

num_of_epochs = 10
data_path = "../logs"

def get_model_name(header: EvalLog) -> str:
    s = header.eval.model
    return s.split("/")[1]

def get_task_name(header: EvalLog) -> str:
    s = header.eval.task
    return s.split("/")[1]

def get_score(score: EvalSampleSummary) -> int:
    assert(score.scores is not None)
    val = score.scores["includes"].value
    if val == "C":
        return 1
    elif val == "I":
        return 0
    raise ValueError(f"Invalid score: {val}")

def read_csv(name: str) -> pd.DataFrame:
    task_name, model_name = name.split("/")
    task = "cybench" if "cybench" in task_name.lower() else "intercode"
    model = "mistral" if "mistral" in model_name.lower() else "gemma"
    df = pd.read_csv(f"../results/final_{model}_{task}_{num_of_epochs}_epochs.csv", index_col=0, header=0).iloc[:-1, :-1]
    return df

def compute_values(df: pd.DataFrame) -> tuple[list, list]:
    """Compute cumulative success rate (mean) and standard deviation across number of attempts."""
    means = []
    stdevs = []
    for e in range(1, num_of_epochs+1):
        sample = df.iloc[:, :e]
        m = sample.mean(axis=1)
        means.append(np.mean(m))
        stdevs.append(np.std(m))
    return means, stdevs

def compute_ci_standard(y, std):
    ci_lower = y - 1.96 * np.array(std) / np.sqrt(len(y))
    ci_upper = y + 1.96 * np.array(std) / np.sqrt(len(y))
    # print("CI", ci_lower, ci_upper)
    return ci_lower, ci_upper

def compute_ci_bootstrap(df: pd.DataFrame, func: Callable, epochs: int = num_of_epochs, n_bootstrap: int = 100, directory: str | None = None, name: str | None = None):
    total_estimates = []
    # mean_stds = []
    ci_lowers = []
    ci_uppers = []

    for e in range(epochs):
        estimates = []
        for _ in range(n_bootstrap):
            attempts = np.random.choice(df.shape[1], size=e+1, replace=True)
            sample = df.iloc[:, attempts]
            means = sample.mean(axis=1)
            estimates.append(func(means))
        total_estimates.append(estimates)
        # mean_stds.append(np.mean(estimates))
        ci_lowers.append(np.percentile(estimates, 2.5))
        ci_uppers.append(np.percentile(estimates, 97.5))

    if directory is not None:
        assert(name is not None)
        task_name, model_name = name.split("/")
        task = "cybench" if "cybench" in task_name.lower() else "intercode"
        model = "mistral" if "mistral" in model_name.lower() else "gemma"
        out = pd.DataFrame(total_estimates, index=range(1, num_of_epochs + 1), columns=range(1, 101))
        out.to_csv(f"{directory}/{model}_{task}_std_estimates.csv")
    # print(name, mean_stds, ci_lowers, ci_uppers)
    return ci_lowers, ci_uppers

