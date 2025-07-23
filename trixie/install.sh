#! /bin/bash

venv_name=""
if [ $# -eq 0 ]; then
    venv_name=".venv"
elif [ $# -eq 1 ]; then
    venv_name=$1
else
    echo "Error: Invalid arguments."
    exit 1
fi

module load python/3.12.1-gcc-11.3.1-7tuhjhr

# Confirm Python version & paths
module list
which python
which pip
python --version

mkdir -p ~/venvs
if [ -d "~/venvs/$venv_name" ]; then
    rm -rf ~/venvs/$venv_name
fi
python3 -m venv ~/venvs/$venv_name
ln -s ~/venvs/$venv_name ~/work/$venv_name
source ~/work/$venv_name/bin/activate

pip install -U pip

pip install inspect_ai \
            git+https://github.com/UKGovernmentBEIS/inspect_evals \
            vllm \
            mistral_common # Mistral-Large-Instruct-2411
            # openai

echo -e "\nSuccessfully installed packages to '~/work/$venv_name'."

