# Investigating the Structural Impact of Chain-of-Thought Prompts on Commonsense Reasoning in Large Language Models (LLMs)

This framework facilitates the investigation of two distinct Chain-of-Thought prompt structures—one enforcing a clear multi-step format and the other employing a conversational, free-flow approach—on the commonsense reasoning performance of large language models. It supports comprehensive evaluation on CommonsenseQA and CoS-E benchmarks under both zero-shot and few-shot conditions, measuring accuracy and logical consistency via Natural Language Inference (NLI) to analyze the interplay between prompt format and model architecture.

## Project Structure

```
.
├── src/
│   ├── datasets/
│   │   └── loader.py           # Dataset loading utilities
│   ├── inference/
│   │   └── infer.py           # Model inference implementation
│   ├── cot_extraction/
│   │   └── extractor.py       # Chain-of-thought extraction
│   ├── evaluation/
│   │   ├── accuracy.py        # Accuracy evaluation
│   │   └── entailment.py      # Entailment ratio evaluation
│   ├── utils/
│   │   └── nli_client.py      # NLI service client
│   ├── main.py                # Zero-shot experiment for CommonsenseQA
│   ├── main_cose_entail.py    # Zero-shot experiment for CoS-E
│   ├── main_csqa_fewshot.py   # Few-shot experiment for CommonsenseQA
│   ├── main_cose_fewshot.py   # Few-shot experiment for CoS-E
│   ├── run_experiments.py     # Zero-shot experiment runner
│   └── run_experiments_few_shot.py  # Few-shot experiment runner
├── prompts/
│   └── templates/
│       ├── templated/         # Structured prompt templates
│       └── naturalistic/      # Natural language prompt templates
└── outputs/                   # Experiment results directory
```

## Features

- Support for multiple datasets:
  - CommonsenseQA
  - CoS-E (Chain of Thought Explanations)
- Multiple prompting strategies:
  - Zero-shot
  - Few-shot
  - Templated prompts
  - Natural language prompts
- Comprehensive evaluation metrics:
  - Accuracy
  - Entailment ratio
- Flexible experiment execution:
  - Sequential execution
  - Parallel execution
  - Single task execution
- Multiple model support:
  - Mistral-7B
  - Falcon3-7B
- Flexible sample size configuration

## Requirements

- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`):
  - torch
  - transformers
  - datasets
  - tqdm
  - numpy
  - pandas
  - scikit-learn

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama:
   - For Windows: Download and install from [Ollama Windows](https://ollama.ai/download/windows)
   - For Linux: 
     ```bash
     curl https://ollama.ai/install.sh | sh
     ```
   - For macOS:
     ```bash
     curl https://ollama.ai/install.sh | sh
     ```

## Dataset Setup

1. Create a `data` directory in the project root:
```bash
mkdir data
```

2. Download CommonsenseQA dataset:
```bash
# Download from Hugging Face
python -c "from datasets import load_dataset; dataset = load_dataset('commonsense_qa'); dataset.save_to_disk('data/commonsense_qa')"
```

3. Download CoS-E dataset:
```bash
# Download from Hugging Face
python -c "from datasets import load_dataset; dataset = load_dataset('cos_e'); dataset.save_to_disk('data/cos_e')"
```

## Model Setup

1. Pull Mistral-7B model:
```bash
ollama pull mistral:7b
```

2. Pull Falcon3-7B model:
```bash
ollama pull falcon3:7b
```

3. Verify model installation:
```bash
# List installed models
ollama list
```

## Usage

### Running Zero-shot Experiments

1. Sequential execution of all zero-shot experiments:
```bash
python -m src.run_experiments --mode sequential [--model MODEL_NAME] [--sample_size SAMPLE_SIZE]
```

2. Parallel execution of all zero-shot experiments:
```bash
python -m src.run_experiments --mode parallel [--model MODEL_NAME] [--sample_size SAMPLE_SIZE]
```

3. Run a single zero-shot experiment:
```bash
python -m src.run_experiments --mode single --dataset commonsenseqa --prompt_type templated --model MODEL_NAME --sample_size SAMPLE_SIZE
```

### Running Few-shot Experiments

1. Sequential execution of all few-shot experiments:
```bash
python -m src.run_experiments_few_shot --mode sequential [--model MODEL_NAME] [--sample_size SAMPLE_SIZE]
```

2. Parallel execution of all few-shot experiments:
```bash
python -m src.run_experiments_few_shot --mode parallel [--model MODEL_NAME] [--sample_size SAMPLE_SIZE]
```

3. Run a single few-shot experiment:
```bash
python -m src.run_experiments_few_shot --mode single --dataset commonsenseqa --prompt_type templated --model MODEL_NAME --sample_size SAMPLE_SIZE
```

### Command Line Arguments

- `--mode`: Execution mode
  - `sequential`: Run experiments sequentially
  - `parallel`: Run experiments in parallel
  - `single`: Run a single experiment
- `--dataset`: Dataset to use (for single mode)
  - `commonsenseqa`
  - `cose`
- `--prompt_type`: Prompt type to use (for single mode)
  - `templated`
  - `natural`
- `--model`: Model to use (optional, default: mistral:7b)
  - `mistral:7b`: Mistral-7B model
  - `falcon3:7b`: Falcon3-7B model
- `--sample_size`: Number of samples to evaluate (optional, default: 103)
  - Integer value specifying the number of samples to evaluate for each dataset

## Output

The framework generates detailed experiment results in the `outputs` directory:

- JSONL files containing detailed results for each experiment
- Summary statistics including:
  - Accuracy
  - Entailment ratio
  - Sample size
  - Execution time

## Customization

### Adding New Datasets

1. Add dataset loading function in `src/datasets/loader.py`
2. Create corresponding prompt templates in `prompts/templates/`
3. Create new experiment script in `src/`

### Adding New Prompt Types

1. Add new prompt templates in `prompts/templates/`
2. Update experiment scripts to support new prompt types
3. Update experiment runners to include new prompt types

### Adding New Models

1. Ensure the model is compatible with the Hugging Face transformers library
2. Add the model to the `--model` choices in the experiment runners
3. Update the model loading logic if necessary

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

[If you use this framework in your research or work, please cite it here. Example:
```
@misc{yourframework,
  author = {Your Name/Organization},
  title = {Investigating the Structural Impact of Chain-of-Thought Prompts on Commonsense Reasoning in Large Language Models (LLMs)},
  year = {Year},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-repo-link}},
}
```
Please fill in your name/organization, year, and repository link as appropriate.] 