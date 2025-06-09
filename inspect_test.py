#! /bin/python3

from inspect_ai import Task, task
from inspect_ai.dataset import example_dataset, json_dataset, csv_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import (chain_of_thought, generate, self_critique)

@task
def test_security_guide():
    return Task(
        dataset=example_dataset("security_guide"),
        solver=[
          chain_of_thought(),
          generate(),
          self_critique()
        ],
        scorer=model_graded_fact()
    )
