# EasyEval: LLM Mathematics Evaluation Framework

EasyEval is a powerful and flexible framework for evaluating Large Language Models (LLMs). It supports multiple datasets, various model providers, and customizable evaluation criteria. It uses an LLM judge for flexibility 

## Features

- **Multiple Dataset Support**: Evaluate models on 10+ different mathematical datasets
- **Custom Datasets**: Load your own JSON-formatted questions
- **Multiple Model Support**: Test OpenAI, Anthropic, DeepInfra, and DeepSeek models
- **Parallel Processing**: Efficiently evaluate multiple questions in batches
- **Detailed Reports**: Get accuracy metrics with confidence intervals
- **Flexible Configuration**: Customize prompts, evaluation criteria, and more
- **LLM-as-Judge**: Uses an LLM to evaluate the accuracy of model responses

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/EasyEval.git
cd EasyEval
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys as environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
export DEEPINFRA_API_KEY="your-deepinfra-key"
```

## Usage

Run the evaluation script:
```bash
python eval.py
```

The interactive CLI will guide you through:
1. Selecting a dataset
2. Choosing a model to evaluate
3. Specifying an evaluator model
4. Setting the number of questions

### Available Datasets

1. Random MATH questions
2. Hard MATH questions
3. Medium MATH questions
4. Competition MATH questions
5. Hard Competition MATH questions
6. GPQA questions
7. Test MATH questions
8. NuminaMath questions
9. MathInstruct questions
10. Olympiad questions
11. Custom questions (from your own JSON file)

### Using Custom Datasets

When selecting option 11, you'll be prompted to provide a path to your custom JSON file. The file should follow this format:

```json
[
  {
    "question": "Your question text here",
    "solution": "The solution here"
  },
  ...
]
```

The system will look for fields named "question" (or "problem"/"prompt") and "solution" (or "answer"/"output").

#### Alternative Dataset Formats

EasyEval is flexible with JSON structures. Here are other formats it supports:

**Format with nested questions:**
```json
{
  "questions": [
    {
      "problem": "What is the derivative of f(x) = x²?",
      "answer": "f'(x) = 2x"
    },
    // more questions...
  ]
}
```

**Format with additional fields:**
```json
[
  {
    "question": "Evaluate the triple integral $\\iiint_{E} dV$ where $E$ is the region bounded by the sphere $\\rho = 3$.",
    "prompt": "Solve step by step, then provide your final answer wrapped in a box.",
    "solution": "$\\boxed{\\frac{36\\pi}{3}}$",
    "answer": "36π/3",
    "difficulty": "medium"
  },
  // more questions...
]
```

If no file is provided, a default example file will be created at `data/custom_questions.json`.

## Customizing Models

The system supports various LLM providers through the `lib/inference.py` file. You can:

1. **Add new models**: Extend the `generate_text()` function to support additional models
2. **Customize prompts**: Modify the prompts in `generate_answer()` function in `eval.py`
3. **Adjust parameters**: Change temperature, max tokens, and other generation parameters

### Supported Models

Currently, EasyEval supports the following model types:

- **OpenAI models**: All GPT models including GPT-4 and custom fine-tuned models (`gpt-4-turbo`, `gpt-3.5-turbo`, etc.)
- **Anthropic models**: Claude models (`claude-3-opus`, `claude-3-sonnet`, etc.)
- **DeepInfra models**: Meta and Qwen models (`meta-llama/Llama-2-70b-chat-hf`, `Qwen/Qwen2-72B-Instruct`, etc.)
- **DeepSeek models**: DeepSeek's AI models (`deepseek-coder`, `deepseek-chat`, etc.)

### Adding a New Model Provider

To add support for a new model provider, edit `lib/inference.py`:

1. Add the required API key
```python
new_provider_api_key = os.getenv('NEW_PROVIDER_API_KEY')
```

2. Initialize the client
```python
new_provider_client = YourProviderClient(api_key=new_provider_api_key)
```

3. Add a new condition in the `generate_text()` function
```python
elif model.startswith("new-provider-"):
    response = await new_provider_client.generate(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.text.strip()
```

## Evaluation Process

The evaluation process follows these steps:

1. **Question Selection**: Questions are selected from the chosen dataset
2. **Answer Generation**: The selected model generates an answer for each question
3. **Answer Evaluation**: The evaluator model compares the generated answer with the ground truth
4. **Metrics Calculation**: Accuracy and confidence intervals are calculated
5. **Results Storage**: Detailed results are saved in the `logs` directory

## LLM-as-Judge Evaluation

EasyEval uses the "LLM-as-Judge" technique for evaluation, where one language model evaluates the responses of another:

### How It Works

1. A primary LLM generates answers to mathematical problems
2. A judge LLM (specified as the "evaluator model") compares these answers to ground truth
3. The judge determines if the answer is "ACCURATE" or "INCORRECT"

### Benefits of LLM-as-Judge

- **Flexible Evaluation**: Understands equivalent mathematical expressions (e.g., π/2 vs 1.57)
- **Reasoning Assessment**: Can evaluate intermediate steps, not just final answers
- **Natural Language Understanding**: Handles varied response formats and extracts key information

### Customizing the Judge

You can customize the judge's behavior in `lib/eval_response.py`:

```python
async def evaluate_text(eval_model: str, modelAnswer: str, groundTruthAnswer: int, temperature: float = 0) -> str:
    prompt = f"""
    You will be given a ground truth answer and a model answer. Please output ACCURATE if the model answer matches the ground truth answer or INCORRECT otherwise. Please only return ACCURATE or INACCURATE. It is very important for my job that you do this. Just check final answer don't check working out. Be flexible with different formats of the model answer e.g decimal, fraction, integer, etc.

    <GroundTruthAnswer>
    {groundTruthAnswer}
    </GroundTruthAnswer>

    <ModelAnswer>
    {modelAnswer}
    </ModelAnswer>
    """
    
    # The judge's response
    isAccurate = await generate_text(
        model=eval_model,
        prompt=prompt,
        max_tokens=10,
        temperature=temperature
    )
    
    # Interpretation of the judge's response
    if "ACCURATE" in isAccurate.strip().upper():
        return 1, 0  # Correct answer
    elif "INCORRECT" in isAccurate.strip().upper():
        return 0, 0  # Incorrect answer
    else:
        return 0, 1  # Bad response from the judge
```

Recommended judge models include:
- `gpt-4-turbo` (OpenAI)
- `claude-3-opus-20240229` (Anthropic)
- `llama-3.1-70b-versatile` (Meta)

## Customizing Evaluation Criteria

You can modify the evaluation criteria by editing `lib/eval_response.py`:

```python
async def evaluate_text(eval_model: str, modelAnswer: str, groundTruthAnswer: int, temperature: float = 0) -> str:
    # Customize the prompt to change evaluation criteria
    prompt = f"""
    You will be given a ground truth answer and a model answer. 
    Please output ACCURATE if the model answer matches the ground truth answer or INCORRECT otherwise.
    """
    ...
```

## Project Structure

- `eval.py`: Main evaluation script
- `lib/inference.py`: Model integration for text generation
- `lib/eval_response.py`: Logic for evaluating model responses
- `lib/get_dataset_question.py`: Dataset loading and question retrieval
- `logs/`: Directory for evaluation results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Environment Variables

EasyEval uses environment variables for API keys and configuration. You can set these directly or use a `.env` file:

### Using a .env File

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
DEEPSEEK_API_KEY=your-deepseek-key
DEEPINFRA_API_KEY=your-deepinfra-key
```

Then load these variables by adding to the top of your scripts:

```python
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env
```

### Required API Keys

Different models require different API keys:

| Model Type | Environment Variable |
|------------|----------------------|
| OpenAI (GPT) | `OPENAI_API_KEY` |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| DeepInfra | `DEEPINFRA_API_KEY` |

## Troubleshooting

### API Key Issues

If you encounter authentication errors:

- Verify that your API keys are correctly set in environment variables
- Check that you haven't exceeded rate limits or your account balance
- Ensure that you have access to the specified models in your subscription

### Model Not Found

If a model is not found:

- Check the spelling of the model name
- Make sure you have access to the model in your subscription
- Verify that the model is supported by the provider

### Custom Dataset Issues

If there are issues with your custom dataset:

- Verify your JSON file is properly formatted and valid
- Check that at least one of the recognized field pairs is present
- Ensure the file path is correct

### Performance Optimization

For better performance:

- Adjust the `BATCH_SIZE` in `eval.py` based on your available resources
- Use smaller models for evaluation if judge model is too slow
- Consider using more deterministic models (lower temperature) for the judge
