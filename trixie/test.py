from vllm import LLM, SamplingParams

# Choose a model that works with vLLM (e.g., mistralai/Mistral-7B-Instruct-v0.2)
model = "mistralai/Mistral-Large-Instruct-2411"

# Set up the model and sampling parameters
llm = LLM(model=model)  # This should use GPU automatically if available
sampling_params = SamplingParams(temperature=0.7, top_p=0.95)

# Run a simple prompt
output = llm.generate("What is the capital of France?", sampling_params)

print("Output:", output[0].outputs[0].text.strip())
