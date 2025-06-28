#! ../.venv/bin/python3

import sys, pathlib
from inspect_ai.log import EvalLog, EvalSample, read_eval_log_sample, read_eval_log
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageTool

from utils import _get_model_name, _get_task_name

def get_transcript(file: str, task: str, epochs: list, directory: str | None = None) -> str | None:
    header: EvalLog = read_eval_log(file, header_only=True)
    model = _get_model_name(header)
    dataset = _get_task_name(header)

    output = ""
    for epoch in epochs:
        print(f"Getting transcript for... {model.capitalize()} - {dataset} - Task: {task} - Epoch {epoch}")
        output += f"{model.capitalize()} - {dataset} - Task: {task} - Epoch {epoch}\n"

        sample: EvalSample = read_eval_log_sample(file, task, int(epoch))
        for i in sample.messages:
            if isinstance(i, ChatMessageSystem):
                assert(isinstance(i.content, str))
                content = i.content.replace("\n", "\n\t")
                output += f"System: {content}\n"
            elif isinstance(i, ChatMessageUser):
                assert(isinstance(i.content, str))
                content = i.content.replace("\n", "\n\t")
                output += f"User: {content}\n"
            elif isinstance(i, ChatMessageAssistant):
                assert(isinstance(i.content, str))
                content = i.content.replace("\n", "\n\t")
                output += f"Assistant: {content}"
                if i.tool_calls is not None:
                    output += f"\n\tTool Calls:"
                    for j in i.tool_calls:
                        if j.function == "bash":
                            output += f"\t\t- {j.function}: `{j.arguments['cmd']}`"
                        else:
                            output += f"\t\t- {j.function}: {j.arguments}"
                output += "\n"
            elif isinstance(i, ChatMessageTool):
                assert(isinstance(i.content, str))
                content = i.content.replace("\n", "\n\t")
                output += f"Tool Response:\n\t{content}\n"
            else:
                output += f"Other: {i}\n"
        
        if directory is not None:
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            with open(f"{directory}/{model.lower()}_{dataset.lower()}_{task}-{epoch}.txt", "w") as f:
                f.write(output)
        return output

if __name__ == "__main__":
    assert(len(sys.argv) >= 4)
    get_transcript(sys.argv[1], sys.argv[2], sys.argv[3:], directory="../transcripts")
