#! /bin/bash

read -p "Is Python environment already set up? [y/N]: " answer

if [[ ! "$answer" =~ ^[yY]$ ]]; then
    # Confirm Python version & paths
    module load python/3.12.1-gcc-11.3.1-7tuhjhr
    module list 
    which python
    which pip
    python --version

    read -p "Name of Python virtual environment: " venv_name
    mkdir -p $HOME/venvs
    if [ -d "$HOME/venvs/$venv_name" ]; then
        rm -rf $HOME/venvs/$venv_name
    fi
    python3 -m venv $HOME/venvs/$venv_name
    ln -s $HOME/venvs/$venv_name $venv_name
    source $venv_name/bin/activate

    pip install -U pip
    pip install inspect_ai \
                git+https://github.com/UKGovernmentBEIS/inspect_evals \
                openai \
                # vllm \
                # mistral_common # Mistral-Large-Instruct-2411
    # pip freeze > requirements.txt
    deactivate
    echo -e "\nSuccessfully installed Python packages to '$venv_name'."
fi

read -p "Is Ollama already set up? [y/N]: " answer

if [[ ! "$answer" =~ ^[yY]$ ]]; then
    # Manual installation of Ollama
    curl -LO https://ollama.com/download/ollama-linux-amd64.tgz
    tar -xzf ollama-linux-amd64.tgz
    mkdir -p $HOME/bin
    mv ollama/bin/ollama $HOME/bin/ollama
    echo -e "\nSuccessfully installed Ollama to '$HOME/bin/ollama'."
fi
