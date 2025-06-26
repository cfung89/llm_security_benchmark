#! /bin/bash

# Setup Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Setup Ollama

# docker pull ollama/ollama
#
# curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
#     | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
#     | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
#     | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
# apt-get update
#
# apt-get install -y nvidia-container-toolkit
#
# nvidia-ctk runtime configure --runtime=docker
# systemctl restart docker
#
# docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
#
# docker exec -it ollama ollama run mistral-large
# docker exec -it ollama ollama run gemma2
