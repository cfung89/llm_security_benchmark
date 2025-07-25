# 0_Testing

## Table of Contents

1. [Installation](#installation)
    1. [Install Python dependencies](#python_deps)
    2. [Install Ollama](#ollama)
2. [Usage](#usage)
    1. [Running Intercode from inspect_evals](#intercode_evals)
    2. [Running Cybench from inspect_evals](#cybench_evals)
    3. [Running Cybench from usnistgov/caisi-cyber-evals](#cybench_caisi)
3. [Results](#results)

## <a name="installation" /> 1. Installation

### <a name="python_deps" /> i. Install Python dependencies

On Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install inspect_ai
pip install git+https://github.com/UKGovernmentBEIS/inspect_evals
pip install openai
```

Another option is to copy the `requirements.txt` file from [https://github.com/cfung89/llm_security_benchmark/blob/main/requirements.txt](https://github.com/cfung89/llm_security_benchmark/blob/main/requirements.txt), which includes additional dependencies.
Then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Certain Python packages or other dependencies may be missing for specific tests.

### <a name="ollama" /> ii. Install Ollama

#### Using Docker

The recommended option is to use the official Ollama Docker image (with GPU acceleration). The instructions are here: [https://hub.docker.com/r/ollama/ollama](https://hub.docker.com/r/ollama/ollama).

>[!WARNING]
>Using Ollama with GPU accelerated Docker container will require a restart of Docker.

Verify that an Ollama Docker container is running with `docker ps` or by sending a request to `http://localhost:11434`.

Once an Ollama Docker container is running, install the required LLM with:
```bash
docker exec -it ollama ollama pull <model-name>
```

#### Other

Ollama can be installed directly on the device: [https://ollama.com/download](https://ollama.com/download).
Then, install the required LLM with:
```bash
ollama pull <model-name>
```

## <a name="usage" /> 2. Usage

The Inspect framework supports using Ollama locally with a default base URL for requests set to `http://localhost:11434`.
To run tests, the arguments are as follows:

```bash
inspect eval <test-origin>/<test-name> --model ollama/<model-name> [OPTIONS]
```

`[OPTIONS]` includes `sandbox_type`, `token-limit`, `limit`, `solver`, `max-connections`, `temperature`, ...

### <a name="intercode_evals" /> i. Running Intercode from inspect_evals

```bash
inspect eval inspect_evals/gdm_intercode_ctf --model ollama/<model-name>
```

### <a name="cybench_evals" /> ii. Running Cybench from inspect_evals

Additional setup is required. Read the documentation at [https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/cybench#security-considerations](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/cybench#security-considerations).

Then, run:
```bash
inspect eval inspect_evals/cybench --model ollama/<model-name> -T sandbox_type=<sandbox-type>
```

### <a name="cybench_caisi" /> iii. Running Cybench from usnistgov/caisi-cyber-evals

>[!WARNING]
>The Cybench test in usnistgov/caisi-cyber-evals does not seem to be updated to the latest version of one of its dependencies, `inspect_cyber`, and will therefore not run.

In another directory/project:

```bash
git clone https://github.com/usnistgov/caisi-cyber-evals.git
cd caisi-cyber-evals
uv venv
source .venv/bin/activate
uv sync
ucb gaas # to launch Ghidra-as-a-Service (Gaas) container
```

Then, to run the test:

```bash
inspect eval ucb/cybench --solver ucb/agent --model ollama/<model-name>
```

## <a name="results" /> 3. Results

In order to view/analyze results, see [1_Analysis](./1_Analysis.md).
