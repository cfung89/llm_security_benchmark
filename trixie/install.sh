#! /bin/bash

py_expanded_dir=""
ollama_expanded_dir=""
project_expanded_dir=""

read -p "Is your project directory set up (in ~/work/...)? [y/N]: " answer
if [[ ! "$answer" =~ ^[yY]$ ]]; then
    # Manual installation of Ollama
    read -p "Path where the Python virtual environment should be saved (e.g.: ~/work): " dir
    project_expanded_dir=$(eval echo "$dir")
    mkdir -p $project_expanded_dir
fi

read -p "Is the Python environment already set up? [y/N]: " answer
if [[ ! "$answer" =~ ^[yY]$ ]]; then
    # Confirm Python version & paths
    module load python/3.12.1-gcc-11.3.1-7tuhjhr
    module list
    python --version

    read -p "Path where the Python virtual environment should be saved (e.g.: ~/venvs/.venv): " venv_name
    py_expanded_dir=$(eval echo "$venv_name")

    mkdir -p $py_expanded_dir
    python3 -m venv $py_expanded_dir
    source $py_expanded_dir

    which python
    which pip

    pip install -U pip
    pip install inspect_ai \
                git+https://github.com/UKGovernmentBEIS/inspect_evals \
                openai \
    deactivate
    echo -e "\nSuccessfully installed Python packages to '$venv_name'."

    echo -e "Symlink $py_expanded_dir to $project_expanded_dir? [y/N]: " answer
    if [[ ! "$answer" =~ ^[yY]$ ]]; then
        ln -s $py_expanded_dir $project_expanded_dir
    fi
fi

read -p "Is Ollama already set up? [y/N]: " answer
if [[ ! "$answer" =~ ^[yY]$ ]]; then
    # Manual installation of Ollama
    read -p "Path where the Python virtual environment should be saved (e.g.: ~/bin/ollama): " dir
    ollama_expanded_dir=$(eval echo "$dir")
    mkdir -p $ollama_expanded_dir
    cd $ollama_expanded_dir

    curl -LO https://ollama.com/download/ollama-linux-amd64.tgz
    tar -xzf ollama-linux-amd64.tgz

    mkdir -p $HOME/bin
    mv ollama/ ollama_temp/
    mv ollama_temp/bin/ollama .
    rm -rf ollama_temp ollama-linux-amd64.tgz

    echo -e "\nSuccessfully installed Ollama to '$ollama_expanded_dir/ollama'."

    echo -e "Symlink $ollama_expanded_dir to $project_expanded_dir? [y/N]: " answer
    if [[ ! "$answer" =~ ^[yY]$ ]]; then
        ln -s $ollama_expanded_dir/ollama $project_expanded_dir
    fi
fi
