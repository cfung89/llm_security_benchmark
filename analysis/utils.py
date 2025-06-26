def _get_model_name(header) -> str:
    s = header.eval.model
    return s.split("/")[1]

def _get_task_name(header) -> str:
    s = header.eval.task
    return s.split("/")[1]
