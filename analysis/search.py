#! .venv/bin/python3

import os
from inspect_ai.log import read_eval_log, read_eval_log_samples
import pandas as pd

from utils import _get_model_name, _get_task_name

data_path = "./data"
files = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f[-5:] == ".eval"]

f = open("output.txt", "w")
f_str = "No such file"
f2 = open("output2.txt", "w")
f_str2 = "does not exist"
for i in files:
    filename = i.replace(data_path, "./results").replace(".eval", ".csv")
    df = pd.read_csv(filename, index_col=0, header=0)
    header = read_eval_log(i, header_only=True)
    model = _get_model_name(header)
    task = _get_task_name(header)
    if task == "cybench":
        continue
    name = f"{task}/{model.capitalize()}"
    print(f"Searching {name}...")

    # summaries = read_eval_log_sample_summaries(i)
    # print(summaries)
    # log = read_eval_log(i)
    samples = read_eval_log_samples(i)
    s_11 = f"{name}\nSearching for string: \"{f_str}\"\nid,epoch\n"
    s_12 = f"{name}\nSearching for string: \"{f_str2}\"\nid,epoch\n"
    s_21 = f"{name}\nSearching for string: \"{f_str}\"\nid,epoch\n"
    s_22 = f"{name}\nSearching for string: \"{f_str2}\"\nid,epoch\n"
    for s in samples:
        if f_str in str(s.events):
            s_11 += f"{s.id},{s.epoch}\n"
        elif f_str2 in str(s.events):
            s_12 += f"{s.id},{s.epoch}\n"
        if int(df.loc[str(s.id), str(s.epoch)]) == 0:
            if f_str in str(s.events):
                s_21 += f"{s.id},{s.epoch}\n"
            elif f_str2 in str(s.events):
                s_22 += f"{s.id},{s.epoch}\n"
    print(f"{s_11}", file=f)
    print(s_12, file=f)
    print(f"{s_21}", file=f2)
    print(s_22, file=f2)
f.close()
f2.close()

# data_path = "./logs"
# files = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f[-5:] == ".json"]
# for i in files:
#     header = read_eval_log(i, header_only=True)
#     model = _get_model_name(header)
#     task = _get_task_name(header)
#     if task == "cybench":
#         continue
#     name = f"{task}/{model.capitalize()}"
#     print(f"Searching {name}...")

#     search(i)
#     print()
