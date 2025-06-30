#! ../.venv/bin/python3

import os, pathlib
from inspect_ai.log import EvalLog, EvalSampleSummary, read_eval_log, read_eval_log_sample_summaries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import *

def extract_results(write: bool = False) -> dict:
    out = {}
    fs = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f[-5:] == ".eval"]
    for i in fs:
        header: EvalLog = read_eval_log(i, header_only=True)
        model = get_model_name(header)
        task = get_task_name(header)

        samples = header.eval.dataset.sample_ids
        assert(samples is not None)
        samples.append("mean per epoch")

        name = f"{task}/{model.capitalize()}"
        print(f"Extracting {name}...")

        df = pd.DataFrame(index=samples, columns=range(1, num_of_epochs+1))
        summaries: list[EvalSampleSummary] = read_eval_log_sample_summaries(i)
        for j in range(len(summaries)):
            s = summaries[j]
            df.at[s.id, s.epoch] = get_score(s)
        df["mean per task"] = df.mean(axis=1)
        df.iloc[-1] = df.mean()
        if write:
            new_name = i.split("/")[-2].replace(".eval", ".csv")
            df.to_csv(f"../results/{new_name}")
        out[name] = df
    return out

def line_plot(data: dict, directory: str = "../results") -> None:
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    x = np.array(range(1, 11)) # number of attempts
    fig1, (cybench_mean, intercode_mean) = plt.subplots(1, 2, figsize=(12, 4)) # mean
    fig2, (cybench_stdev, intercode_stdev) = plt.subplots(1, 2, figsize=(12, 4)) # stdev
    fig3, (cybench_mean_ci, intercode_mean_ci) = plt.subplots(1, 2, figsize=(12, 4)) # mean with CI
    fig4, (cybench_stdev_ci, intercode_stdev_ci) = plt.subplots(1, 2, figsize=(12, 4)) # stdev with CI

    for name, df in list(data.items()):
        task_name, model_name = name.split("/")
        means, stdevs = compute_values(df)
        print(name, f"Range of mean: {round((max(means) - min(means)) * 100, 2)}%", f"Range of mean: {round((max(stdevs) - min(stdevs)) * 100, 2)}%")
        # mean_ci_lower, mean_ci_upper = compute_ci_standard(means, stdevs)
        mean_ci_lower, mean_ci_upper = compute_ci_bootstrap(df, np.mean)
        std_ci_lower, std_ci_upper = compute_ci_bootstrap(df, np.std)
        colour = ("g", "green") if "mistral" in model_name.lower() else ("b", "blue")

        if task_name == "cybench":
            cybench_mean.plot(x, means, f"{colour[0]}o-", label=model_name)
            cybench_stdev.plot(x, stdevs, f"{colour[0]}o-", label=model_name)

            cybench_mean_ci.plot(x, means, f"{colour[0]}o-", label=model_name)
            cybench_mean_ci.fill_between(x, mean_ci_lower, mean_ci_upper, color=f"light{colour[1]}")
            cybench_mean_ci.plot(x, mean_ci_lower, f"{colour[0]}--")
            cybench_mean_ci.plot(x, mean_ci_upper, f"{colour[0]}--")

            cybench_stdev_ci.plot(x, stdevs, f"{colour[0]}o-", label=model_name)
            cybench_stdev_ci.fill_between(x, std_ci_lower, std_ci_upper, color=f"light{colour[1]}")
            cybench_stdev_ci.plot(x, std_ci_lower, f"{colour[0]}--")
            cybench_stdev_ci.plot(x, std_ci_upper, f"{colour[0]}--")

        elif task_name == "gdm_intercode_ctf":
            intercode_mean.plot(x, means, f"{colour[0]}o-", label=model_name)
            intercode_stdev.plot(x, stdevs, f"{colour[0]}o-", label=model_name)

            intercode_mean_ci.plot(x, means, f"{colour[0]}o-", label=model_name)
            intercode_mean_ci.fill_between(x, mean_ci_lower, mean_ci_upper, color=f"light{colour[1]}")
            intercode_mean_ci.plot(x, mean_ci_lower, f"{colour[0]}--")
            intercode_mean_ci.plot(x, mean_ci_upper, f"{colour[0]}--")

            intercode_stdev_ci.plot(x, stdevs, f"{colour[0]}o-", label=model_name)
            intercode_stdev_ci.fill_between(x, std_ci_lower, std_ci_upper, color=f"light{colour[1]}")
            intercode_stdev_ci.plot(x, std_ci_lower, f"{colour[0]}--")
            intercode_stdev_ci.plot(x, std_ci_upper, f"{colour[0]}--")
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    cybench_mean.set_xlabel("Number of attempts")
    intercode_mean.set_xlabel("Number of attempts")
    cybench_stdev.set_xlabel("Number of attempts")
    intercode_stdev.set_xlabel("Number of attempts")
    cybench_mean_ci.set_xlabel("Number of attempts")
    intercode_mean_ci.set_xlabel("Number of attempts")
    cybench_stdev_ci.set_xlabel("Number of attempts")
    intercode_stdev_ci.set_xlabel("Number of attempts")

    cybench_mean.set_ylabel("Success rate")
    intercode_mean.set_ylabel("Success rate")
    cybench_stdev.set_ylabel("Standard deviation (Success rate)")
    intercode_stdev.set_ylabel("Standard deviation (Success rate)")
    cybench_mean_ci.set_ylabel("Success rate")
    intercode_mean_ci.set_ylabel("Success rate")
    cybench_stdev_ci.set_ylabel("Standard deviation (Success rate)")
    intercode_stdev_ci.set_ylabel("Standard deviation (Success rate)")

    cybench_mean.set_title("Cybench")
    intercode_mean.set_title("IntercodeCTF")
    cybench_stdev.set_title("Cybench")
    intercode_stdev.set_title("IntercodeCTF")
    cybench_mean_ci.set_title("Cybench")
    intercode_mean_ci.set_title("IntercodeCTF")
    cybench_stdev_ci.set_title("Cybench")
    intercode_stdev_ci.set_title("IntercodeCTF")

    fig1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12, frameon=False)
    fig2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12, frameon=False)
    fig3.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12, frameon=False)
    fig4.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12, frameon=False)

    fig1.suptitle("Success Rate over Number of Attempts (Magnified)")
    fig2.suptitle("Standard Deviation of Success Rate over Number of Attempts (Magnified)")
    fig3.suptitle("Success Rate over Number of Attempts (Magnified)")
    fig4.suptitle("Standard Deviation of Success Rate over Number of Attempts (Magnified)")

    fig1.tight_layout(rect=(0, 0.05, 1, 1))
    fig2.tight_layout(rect=(0, 0.05, 1, 1))
    fig3.tight_layout(rect=(0, 0.05, 1, 1))
    fig4.tight_layout(rect=(0, 0.05, 1, 1))

    fig1.savefig(f"{directory}/mean_zoom.png")
    fig2.savefig(f"{directory}/stdev_zoom.png")
    fig3.savefig(f"{directory}/mean_zoom_ci.png")
    fig4.savefig(f"{directory}/stdev_zoom_ci.png")

    fig1.suptitle("Success Rate over Number of Attempts")
    fig2.suptitle("Standard Deviation of Success Rate over Number of Attempts")
    fig3.suptitle("Success Rate over Number of Attempts")
    fig4.suptitle("Standard Deviation of Success Rate over Number of Attempts")

    cybench_mean.set_ylim(0, 1)
    intercode_mean.set_ylim(0, 1)
    cybench_stdev.set_ylim(0, 1)
    intercode_stdev.set_ylim(0, 1)
    cybench_mean_ci.set_ylim(0, 1)
    intercode_mean_ci.set_ylim(0, 1)
    cybench_stdev_ci.set_ylim(0, 1)
    intercode_stdev_ci.set_ylim(0, 1)

    fig1.savefig(f"{directory}/mean.png")
    fig2.savefig(f"{directory}/stdev.png")
    fig3.savefig(f"{directory}/mean_ci.png")
    fig4.savefig(f"{directory}/stdev_ci.png")
    return

def box_plot(data: dict, directory: str = "../results") -> None:
    cybench_ls, cybench_labels = [], []
    intercode_ls, intercode_labels = [], []

    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    fig, (cybench, intercode) = plt.subplots(1, 2, figsize=(12, 4))
    for name, df in list(data.items()):
        task_name, model_name = name.split("/")
        if task_name == "cybench":
            cybench_ls.append(df[0])
            cybench_labels.append(model_name)
        elif task_name == "gdm_intercode_ctf":
            intercode_ls.append(df[0])
            intercode_labels.append(model_name)
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    cybench.boxplot(cybench_ls, tick_labels=cybench_labels)
    intercode.boxplot(intercode_ls, tick_labels=intercode_labels)

    cybench.set_ylabel("Success rate")
    intercode.set_ylabel("Success rate")
    cybench.set_title("Cybench")
    intercode.set_title("IntercodeCTF")
    fig.suptitle("Box plot (Magnified)")

    fig.savefig(f"{directory}/boxplot_zoom.png")

    cybench.set_ylim(0, 1)
    intercode.set_ylim(0, 1)
    fig.suptitle("Box plot")

    fig.savefig(f"{directory}/boxplot.png")
    return

if __name__ == "__main__":
    data = extract_results(write=True)
    line_plot(data)
    # box_plot(data)
