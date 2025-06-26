#! ../.venv/bin/python3

import sys
from inspect_ai.log import read_eval_log_sample, read_eval_log
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageTool

from utils import _get_model_name, _get_task_name

assert(len(sys.argv) >= 4)
file = sys.argv[1]
task = sys.argv[2]

if __name__ == "__main__":
    header = read_eval_log(file, header_only=True)
    model = _get_model_name(header)
    dataset = _get_task_name(header)
    for epoch in sys.argv[3:]:
        print(f"Extracting... {model.capitalize()} - {dataset} - Task: {task} - Epoch {epoch}")
        f = open(f"../transcripts/{model.lower()}_{dataset.lower()}_{task}-{epoch}.txt", "w")
        print(f"{model.capitalize()} - {dataset} - Task: {task} - Epoch {epoch}\n", file=f)

        sample = read_eval_log_sample(file, task, int(epoch))
        for i in sample.messages:
            if isinstance(i, ChatMessageSystem):
                assert(isinstance(i.content, str))
                content = i.content.replace("\n", "\n\t")
                print(f"System: {content}", file=f)
                print(file=f)
            elif isinstance(i, ChatMessageUser):
                assert(isinstance(i.content, str))
                content = i.content.replace("\n", "\n\t")
                print(f"User: {content}", file=f)
                print(file=f)
            elif isinstance(i, ChatMessageAssistant):
                assert(isinstance(i.content, str))
                content = i.content.replace("\n", "\n\t")
                print(f"Assistant: {content}", file=f)
                if i.tool_calls is not None:
                    print(f"\n\tTool Calls:", file=f)
                    for j in i.tool_calls:
                        if j.function == "bash":
                            print(f"\t\t- {j.function}: `{j.arguments['cmd']}`", file=f)
                        else:
                            print(f"\t\t- {j.function}: {j.arguments}", file=f)
                print(file=f)
            elif isinstance(i, ChatMessageTool):
                assert(isinstance(i.content, str))
                content = i.content.replace("\n", "\n\t")
                print(f"Tool Response:\n\t{content}", file=f)
                print(file=f)
            else:
                print(f"Other: {i}", file=f)
                print(file=f)
        f.close()

