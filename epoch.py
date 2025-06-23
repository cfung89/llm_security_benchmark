#! .venv/bin/python3

import os
from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries
import matplotlib.pyplot as plt
import numpy as np

num_of_epochs = 10
data_path = "./data"

def extract() -> dict:
    out = {}
    fs = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f[-5:] == ".eval"]
    for i in fs:
        header = read_eval_log(i, header_only=True)
        model = _get_model_name(header)
        task = _get_task_name(header)
        name = f"{task}/{model.capitalize()}"
        print(f"Extracting {name}...")

        summaries = read_eval_log_sample_summaries(i)
        total = 0
        task_output = [0] * num_of_epochs
        for j in range(len(summaries)):
            s = summaries[j]
            task_output[s.epoch-1] += _get_score(s)
            total += 1
            # print(s.id, s.epoch, s.scores["includes"].value)
        out[name] = (task_output, total/10)
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
    for name, d in list(data.items()):
        task_name, model_name = name.split("/")
        ci_lower, ci_upper = _compute_ci(d[1])
        colour = ("g", "green") if "mistral" in model_name.lower() else ("b", "blue")
        if task_name == "cybench":
            cybench1.plot(x, d[0], f"{colour[0]}o-", label=model_name)
            cybench2.plot(x, d[1], f"{colour[0]}o-", label=model_name)
            cybench2.fill_between(x, ci_lower, ci_upper, color=f"light{colour[1]}")
        elif task_name == "gdm_intercode_ctf":
           intercode1.plot(x, d[0], f"{colour[0]}o-", label=model_name)
            intercode2.plot(x, d[1], f"{colour[0]}o-", label=model_name)
            intercode2.fill_between(x, ci_lower, ci_upper, color=f"light{colour[1]}")
        else:
            print(model_name)
            raise ValueError

    cybench1.set_xlabel("Epochs")
    intercode1.set_xlabel("Epochs")
    cybench2.set_xlabel("Epochs")
    intercode2.set_xlabel("Epochs")

    cybench1.set_ylabel("Mean")
    intercode1.set_ylabel("Mean")
    cybench2.set_ylabel("Stdev")
    intercode2.set_ylabel("Stdev")

    cybench1.set_title("Cybench")
    cybench2.set_title("Cybench")
    intercode1.set_title("IntercodeCTF")
    intercode2.set_title("IntercodeCTF")

    fig1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12, frameon=False)
    fig2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=12, frameon=False)

    fig1.suptitle("Figure 1: Mean per Epoch")
    fig2.suptitle("Figure 2: Standard Deviation per Epoch")

    fig1.tight_layout(rect=[0, 0.05, 1, 1])
    fig2.tight_layout(rect=[0, 0.05, 1, 1])

    fig1.savefig("results/mean_zoom.png")
    fig2.savefig("results/stdev_zoom.png")

    cybench1.set_ylim(0, 1)
    cybench2.set_ylim(0, 1)
    intercode1.set_ylim(0, 1)
    intercode2.set_ylim(0, 1)

    fig1.savefig("results/mean.png")
    fig2.savefig("results/stdev.png")

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
            raise ValueError
    cybench.boxplot(cybench_ls, tick_labels=cybench_labels)
    intercode.boxplot(intercode_ls, tick_labels=intercode_labels)

   cybench.set_title("Cybench")
   intercode.set_title("IntercodeCTF")
   cybench.set_ylabel("Mean")
   intercode.set_ylabel("Mean")
   fig.suptitle("Figure 3: Box plot")
   fig.savefig("results/boxplot_zoom.png")

   cybench.set_ylim(0, 1)
   intercode.set_ylim(0, 1)
   fig.savefig("results/boxplot.png")
   return

def _get_model_name(header) -> str:
    s = header.eval.model
    return s.split("/")[1]

def _get_task_name(header) -> str:
    s = header.eval.task
    return s.split("/")[1]

def _get_score(score) -> float:
    val = score.scores["includes"].value
    if val == "C":
        return 1.0
    elif val == "I":
        return 0.0
    print("INVALID SCORE")
    return float(val)

def _compute_ci(y):
    ci_lower = y - 1.96 * np.std(y) / np.sqrt(len(y))
    ci_upper = y + 1.96 * np.std(y) / np.sqrt(len(y))
    print("CI", ci_lower, ci_upper)
    return ci_lower, ci_upper

# def _compute_ci_bootstrap(data, epoch, n_bootstrap=100):
#     estimates = []
#     for _ in range(n_bootstrap):
#         sample = np.random.choice(data, size=epoch, replace=True)
#         estimates.append(np.mean(sample))
#     ci_lower = y - np.std(estimates)
#     ci_upper = y + np.std(estimates)
#     return ci_lower, ci_upper

if __name__ == "__main__":
    data = extract()
    print("\nDATA")
    print(data)
    out = analyze(data)
    for i in list(out.items()):
        print(i)
    line_plot(out)
    box_plot(out)
