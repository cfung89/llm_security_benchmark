# Trixie

## Scripts

`install.sh` is an installation script which installs Ollama and the Python environments in a specified directory. These files/directories can then be symlinked into `/gpfs/work/${USER}/...`.
`schedule.sh` is an sample script for scheduling a job with this setup.

## Model providers (Ollama)

Ollama must be installed on Trixie manually, as users do not have the permission to set up a GPU accelerated Docker container.

However, at the moment, even with the manual installation, Ollama is unable to retrieve files from the Ollama registry server. The site is in the list of sites blocked by the firewall due to being potential AI services or chat applications. A request was sent to unblock this registry.

A working alternative is to install GGUF files of models from Hugging Face and adding this model as a custom model in Ollama. Manually calling the Ollama command and sending a prompt to the model works with this method.

Example:
```bash
pip install huggingface-hub
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
echo "FROM ./mistral-7b-instruct-v0.2.Q4_K_M.gguf" > Modelfile
ollama create mistral_large_custom -f Modelfile
ollama list
```

## Benchmarks

With security and permission restrictions, it is currently not possible to host the sandbox Docker containers used for Cybench and Intercode as it requires Docker Engine to be installed.
A possible solution could be to host the containers elsewhere and communicate with the model on Beatrix. However, this solution may be blocked due to strict network restrictions on the HPC server.
