# 1_Analysis

## Table of Contents

1. [Viewing Results](#view_results)
    1. [Inspect](#inspect_view)
    2. [JSON transcripts](#json_transcripts)
    3. [Text transcripts](#text_transcripts)
2. [Analysis](#analysis)

## <a name="view_results" /> 1. Viewing Results

By default, eval logs are written to the `./logs` sub-directory of the current working directory.

### <a name="inspect_view" /> i. Inspect

The recommended way of viewing Inspect logs is using its [VSCode Extension](https://inspect.aisi.org.uk/vscode.html).
An alternative is to display the logs in the browser by running `inspect view` in the terminal. The default URL is `localhost:7575`

### <a name="json_transcripts" /> ii. JSON transcripts

In order to convert the `.eval` files in the `./logs` sub-directory to readable text, these files can be converted to equivalent JSON files containing the entire transcript, as well as all the metadata from the test.

```bash
inspect log convert --to json --output-dir ./output ./logs
```

### <a name="text_transcripts" /> iii. Text transcripts

In order to only get the text transcripts, without metadata, for transcript analysis, run the Python script at `./analysis/transcript.py`:

```bash
cd analysis
./transcript.py <eval_filepath> <task_name_or_number> <epoch_num_1> <epoch_num_2> ...
```

Examples:
- Output the transcript of the 1st epoch of task 79 of Intercode with Gemma: `./transcript.py ../logs/final_gemma_intercode_10_epochs.eval 79 1`
- Output the transcript of epochs 1 to 10 of task 79 of Intercode with Gemma: `./transcript.py ../logs/final_gemma_intercode_10_epochs.eval 79 {1..10}`
