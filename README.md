# LLM Security Testing

## Installation

```bash
git clone https://github.com/cfung89/llm_security_benchmark.git

# Setup Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See [Installation notes](./notes/0_Testing.md).

## Usage

### Analysis

#### Epoch analysis

In order to quantitatively analyze the stability of the success rate, we generated line plots representing the cumulative success rate, cumulative standard deviation, and their respective bootstrapped 95% confidence intervals.
Box plots can also be generated.

Run `cd analysis && ./epoch.py` to generate the plots.

#### Transcripts

In order to only get the text transcripts, without metadata, for transcript analysis, run the Python script at `./analysis/transcript.py`:

```bash
cd analysis
./transcript.py <eval_filepath> <task_name_or_number> <epoch_num_1> <epoch_num_2> ...
```

Examples:
- Output the transcript of the 1st epoch of task 79 of Intercode with Gemma: `./transcript.py ../logs/final_gemma_intercode_10_epochs.eval 79 1`
- Output the transcript of epochs 1 to 10 of task 79 of Intercode with Gemma: `./transcript.py ../logs/final_gemma_intercode_10_epochs.eval 79 {1..10}`
