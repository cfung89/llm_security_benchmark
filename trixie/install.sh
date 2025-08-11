#! /bin/bash

read -p "Is the Python environment already set up? [y/N]: " answer

if [[ ! "$answer" =~ ^[yY]$ ]]; then
    # Confirm Python version & paths
    module load python/3.12.1-gcc-11.3.1-7tuhjhr
    module list
    python --version

    read -p "Path where the Python virtual environment should be saved (e.g.: ~/venvs/.venv): " venv_name
    expanded_dir=$(eval echo "$venv_name")

    mkdir -p $expanded_dir
    python3 -m venv $expanded_dir
    source $expanded_dir

    which python
    which pip

    pip install -U pip
    pip install inspect_ai \
                git+https://github.com/UKGovernmentBEIS/inspect_evals \
                openai \
    deactivate
    echo -e "\nSuccessfully installed Python packages to '$venv_name'."
fi

read -p "Is Ollama already set up? [y/N]: " answer

if [[ ! "$answer" =~ ^[yY]$ ]]; then
    # Manual installation of Ollama
    read -p "Path where the Python virtual environment should be saved (e.g.: ~/bin/ollama): " dir
    expanded_dir=$(eval echo "$dir")
    mkdir -p $expanded_dir
    cd $expanded_dir

    curl -LO https://ollama.com/download/ollama-linux-amd64.tgz
    tar -xzf ollama-linux-amd64.tgz

    mkdir -p $HOME/bin
    mv ollama/ ollama_temp/
    mv ollama_temp/bin/ollama .
    rm -rf ollama_temp ollama-linux-amd64.tgz

    echo -e "\nSuccessfully installed Ollama to '$expanded_dir'."
fi
