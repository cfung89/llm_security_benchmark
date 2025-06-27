#! ../.venv/bin/python3

import os, sys
from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import _get_model_name, _get_task_name

num_of_epochs = 10
data_path = "../logs"

def extract() -> dict:
    out = {}
    fs = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f[-5:] == ".eval"]
    for i in fs:
        header = read_eval_log(i, header_only=True)
        samples = header.eval.dataset.sample_ids
        assert(samples is not None)
        samples.append("mean per epoch")
        model = _get_model_name(header)
        task = _get_task_name(header)
        name = f"{task}/{model.capitalize()}"
        print(f"Extracting {name}...")

        df = pd.DataFrame(index=header.eval.dataset.sample_ids, columns=range(1, num_of_epochs+1))
        task_output = [0.0] * num_of_epochs
        summaries = read_eval_log_sample_summaries(i)
        for j in range(len(summaries)):
            s = summaries[j]
            score = _get_score(s)
            task_output[s.epoch-1] += score
            df.at[s.id, s.epoch] = int(score)
            # print(df.head)
            # print(s.id, s.epoch, s.scores["includes"].value)
        out[name] = (task_output, header.eval.dataset.samples)
        new_name = i.split("/")[-1].replace(".eval", ".csv")
        df["mean per task"] = df.mean(axis=1)
        mean_row = df.mean()
        df.iloc[-1] = mean_row
        # df.iloc[-1, -1] = df.iloc[:-1, :-1].mean().mean()
        # print(df.head)
        df.to_csv(f"../results/{new_name}")
    return out

def analyze(d: dict) -> dict:
    out = {}
    for j, data in list(d.items()):
        mean = [i/data[1] for i in data[0]]
        stdev = [((i/data[1])*(1-(i/data[1])))**(0.5) for i in data[0]] # Bernoulli distribution
        out[j] = (np.array(mean), np.array(stdev))
    return out

def line_plot(data: dict) -> None:
    x = np.array(range(1, 11))
    fig1, (cybench1, intercode1) = plt.subplots(1, 2, figsize=(12, 4))
    fig2, (cybench2, intercode2) = plt.subplots(1, 2, figsize=(12, 4))
    fig3, (cybench3, intercode3) = plt.subplots(1, 2, figsize=(12, 4))
    fig4, (cybench4, intercode4) = plt.subplots(1, 2, figsize=(12, 4))

    for name, d in list(data.items()):
        task_name, model_name = name.split("/")
        # _compute_ci(d[0])
        # _compute_ci(d[1])
        means, stdevs = _compute_mean(name)
        mean_stds, ci_lower4, ci_upper4 = _compute_ci_bootstrap(name)
        ci_lower3, ci_upper3 = _compute_ci(means, stdevs)
        colour = ("g", "green") if "mistral" in model_name.lower() else ("b", "blue")
        if task_name == "cybench":
            cybench1.plot(x, means, f"{colour[0]}o-", label=model_name)
            cybench2.plot(x, stdevs, f"{colour[0]}o-", label=model_name)

            cybench3.plot(x, means, f"{colour[0]}o-", label=model_name)
            cybench3.fill_between(x, ci_lower3, ci_upper3, color=f"light{colour[1]}")

            # cybench4.plot(x, d[1], f"{colour[0]}o-", label=model_name)
            # cybench4.fill_between(x, d[1] - ci_lower4, d[1] + ci_upper4, color=f"light{colour[1]}")
            cybench4.plot(x, stdevs, f"{colour[0]}o-", label=model_name)
            cybench4.fill_between(x, ci_lower4, ci_upper4, color=f"light{colour[1]}")
        elif task_name == "gdm_intercode_ctf":
            intercode1.plot(x, means, f"{colour[0]}o-", label=model_name)
            intercode2.plot(x, stdevs, f"{colour[0]}o-", label=model_name)

            intercode3.plot(x, means, f"{colour[0]}o-", label=model_name)
            intercode3.fill_between(x, ci_lower3, ci_upper3, color=f"light{colour[1]}")

            intercode4.plot(x, stdevs, f"{colour[0]}o-", label=model_name)
            intercode4.fill_between(x, ci_lower4, ci_upper4, color=f"light{colour[1]}")
        else:
            raise ValueError(f"Invalid model name: {model_name}")

    cybench1.set_xlabel("Number of attempts")
    intercode1.set_xlabel("Number of attempts")
    cybench2.set_xlabel("Number of attempts")
    intercode2.set_xlabel("Number of attempts")
    cybench3.set_xlabel("Number of attempts")
    intercode3.set_xlabel("Number of attempts")
    cybench4.set_xlabel("Number of attempts")
    intercode4.set_xlabel("Number of attempts")

    cybench1.set_ylabel("Success rate")
    intercode1.set_ylabel("Success rate")
    cybench2.set_ylabel("Standard deviation (Success rate)")
    intercode2.set_ylabel("Standard deviation (Success rate)")
    cybench3.set_ylabel("Success rate")
    intercode3.set_ylabel("Success rate")
    cybench4.set_ylabel("Standard deviation (Success rate)")
    intercode4.set_ylabel("Standard deviation (Success rate)")

    cybench1.set_title("Cybench")
    intercode1.set_title("IntercodeCTF")
    cybench2.set_title("Cybench")
    intercode2.set_title("IntercodeCTF")
    cybench3.set_title("Cybench")
    intercode3.set_title("IntercodeCTF")
    cybench4.set_title("Cybench")
    intercode4.set_title("IntercodeCTF")

    fig1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12, frameon=False)
    fig2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12, frameon=False)
    fig3.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12, frameon=False)
    fig4.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12, frameon=False)

    fig1.suptitle("Figure 1: Success Rate over Number of Attempts (Magnified)")
    fig2.suptitle("Figure 2: Standard Deviation of Success Rate over Number of Attempts (Magnified)")
    fig3.suptitle("Figure 1: Success Rate over Number of Attempts (Magnified)")
    fig4.suptitle("Figure 2: Standard Deviation of Success Rate over Number of Attempts (Magnified)")

    fig1.tight_layout(rect=(0, 0.05, 1, 1))
    fig2.tight_layout(rect=(0, 0.05, 1, 1))
    fig3.tight_layout(rect=(0, 0.05, 1, 1))
    fig4.tight_layout(rect=(0, 0.05, 1, 1))

    fig1.savefig("../results/mean_zoom.png")
    fig2.savefig("../results/stdev_zoom.png")
    fig3.savefig("../results/mean_zoom_ci.png")
    fig4.savefig("../results/stdev_zoom_ci.png")

    fig1.suptitle("Figure 1: Success Rate over Number of Attempts")
    fig2.suptitle("Figure 2: Standard Deviation of Success Rate over Number of Attempts")
    fig3.suptitle("Figure 1: Success Rate over Number of Attempts")
    fig4.suptitle("Figure 2: Standard Deviation of Success Rate over Number of Attempts")

    cybench1.set_ylim(0, 1)
    intercode1.set_ylim(0, 1)
    cybench2.set_ylim(0, 1)
    intercode2.set_ylim(0, 1)
    cybench3.set_ylim(0, 1)
    intercode3.set_ylim(0, 1)
    cybench4.set_ylim(0, 1)
    intercode4.set_ylim(0, 1)

    fig1.savefig("../results/mean.png")
    fig2.savefig("../results/stdev.png")
    fig3.savefig("../results/mean_ci.png")
    fig4.savefig("../results/stdev_ci.png")
    return

def box_plot(data: dict) -> None:
    fig, (cybench, intercode) = plt.subplots(1, 2, figsize=(12, 4))
    cybench_ls = []
    cybench_labels = []
    intercode_ls = []
    intercode_labels = []
    for name, d in list(data.items()):
        task_name, model_name = name.split("/")
        if task_name == "cybench":
            cybench_ls.append(d[0])
            cybench_labels.append(model_name)
        elif task_name == "gdm_intercode_ctf":
            intercode_ls.append(d[0])
            intercode_labels.append(model_name)
        else:
            print(model_name)
            raise ValueError(f"Invalid model name: {model_name}")
    cybench.boxplot(cybench_ls, tick_labels=cybench_labels)
    intercode.boxplot(intercode_ls, tick_labels=intercode_labels)

    cybench.set_title("Cybench")
    intercode.set_title("IntercodeCTF")
    cybench.set_ylabel("Mean")
    intercode.set_ylabel("Mean")
    fig.suptitle("Figure 5: Box plot")
    fig.savefig("../results/boxplot_zoom.png")

    cybench.set_ylim(0, 1)
    intercode.set_ylim(0, 1)
    fig.savefig("../results/boxplot.png")
    return

def _get_score(score) -> float:
    val = score.scores["includes"].value
    if val == "C":
        return 1.0
    elif val == "I":
        return 0.0
    raise ValueError(f"Invalid score: {val}")

def _compute_mean(name):
    task_name, model_name = name.split("/")
    task = "cybench" if "cybench" in task_name.lower() else "intercode"
    model = "mistral" if "mistral" in model_name.lower() else "gemma"
    df = pd.read_csv(f"../results/final_{model}_{task}_{num_of_epochs}_epochs.csv", index_col=0, header=0).iloc[:-1, :-1]

    means = []
    stdevs = []
    for e in range(1, num_of_epochs+1):
        sample = df.iloc[:, :e]
        m = sample.mean(axis=1)
        means.append(np.mean(m))
        stdevs.append(np.std(m))
    return means, stdevs

def _compute_ci(y, std):
    print(std)
    ci_lower = y - 1.96 * np.array(std) / np.sqrt(len(y))
    ci_upper = y + 1.96 * np.array(std) / np.sqrt(len(y))
    # print("CI", ci_lower, ci_upper)
    return ci_lower, ci_upper

def _compute_ci_bootstrap(name, epoch=num_of_epochs, n_bootstrap=100):
    task_name, model_name = name.split("/")
    task = "cybench" if "cybench" in task_name.lower() else "intercode"
    model = "mistral" if "mistral" in model_name.lower() else "gemma"
    df = pd.read_csv(f"../results/final_{model}_{task}_{num_of_epochs}_epochs.csv", index_col=0, header=0).iloc[:-1, :-1]

    mean_stds = []
    ci_lowers = []
    ci_uppers = []
    for e in range(num_of_epochs):
        estimates = []
        for _ in range(n_bootstrap):
            attempts = np.random.choice(df.shape[1], size=e+1, replace=True)
            sample = df.iloc[:, attempts]
            means = sample.mean(axis=1)
            estimates.append(np.std(means))
        mean_stds.append(np.mean(estimates))
        ci_lowers.append(np.percentile(estimates, 2.5))
        ci_uppers.append(np.percentile(estimates, 97.5))
    # print(name, mean_stds, ci_lowers, ci_uppers)
    return mean_stds, ci_lowers, ci_uppers

if __name__ == "__main__":
    data = extract()
    # print("\nDATA")
    # print(data)
    out = analyze(data)
    # for i in list(out.items()):
    #     print(i)
    line_plot(out)
    # box_plot(out)
