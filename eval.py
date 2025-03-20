import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import asyncio
import time
import random
from typing import List, Dict, Any, Tuple, Optional
import math
from lib.inference import generate_text
from lib.eval_response import evaluate_text
import lib.get_dataset_question as datasets

# Configuration
BATCH_SIZE = 100
LOGS_FOLDER = "logs"

async def generate_answer(model: str, problem: str) -> str:
    """Generate an answer using the specified model for a given problem."""
    prompt = f"Solve the following problem. Show your work and clearly indicate your final answer.\n\n{problem}"
    return await generate_text(model=model, prompt=prompt)

async def process_question(model: str, evaluator_model: str, problem: str, solution: str) -> Dict[str, Any]:
    """Process a single question with the model and evaluate the response."""
    try:
        # Generate model's answer
        model_answer = await generate_answer(model, problem)
        
        # Evaluate the answer
        accuracy, bad_response = await evaluate_text(
            eval_model=evaluator_model,
            modelAnswer=model_answer,
            groundTruthAnswer=solution
        )
        
        return {
            "problem": problem,
            "solution": solution,
            "model_answer": model_answer,
            "is_accurate": accuracy == 1,
            "accuracy": accuracy,
            "bad_response": bad_response
        }
    except Exception as e:
        return {
            "problem": problem,
            "solution": solution,
            "model_answer": f"ERROR: {str(e)}",
            "is_accurate": False,
            "accuracy": 0,
            "bad_response": 1,
            "error": str(e)
        }

async def evaluate_batch(model: str, evaluator_model: str, question_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Evaluate a batch of questions and return the results."""
    print(f"Evaluating batch of {len(question_pairs)} questions...")
    start_time = time.time()
    
    # Process questions in parallel
    tasks = [
        process_question(model, evaluator_model, problem, solution)
        for problem, solution in question_pairs
    ]
    results = await asyncio.gather(*tasks)
    
    # Process batch results
    accurate_count = sum(result['accuracy'] for result in results)
    bad_responses = sum(result['bad_response'] for result in results)
    errors = sum(1 for result in results if 'error' in result)
    
    return {
        "results": results,
        "accurate_count": accurate_count,
        "bad_responses": bad_responses,
        "errors": errors,
        "time_taken": time.time() - start_time
    }

def get_user_input() -> Tuple[str, str, str, int, Optional[str]]:
    """Get user input for evaluation settings."""
    print("\n=== LLM Evaluation System ===")
    
    # Get dataset type using the abstracted function
    dataset_options = datasets.get_dataset_options()
    
    print("\nAvailable datasets:")
    for key, value in dataset_options.items():
        print(f"{key}. {value}")
    
    # Add a special note about the custom dataset option
    print("\nNote: For custom dataset (option 11), you'll provide a path to a JSON file containing questions.")
    print("The JSON should have fields like 'question'/'problem' and 'solution'/'answer'.")
    
    dataset_choice = input("\nSelect dataset (1-11): ").strip()
    
    # Get custom filepath if custom JSON dataset is selected
    custom_filepath = None
    if dataset_choice == "11":
        print("\nCustom JSON format example:")
        print("""[
  {
    "question": "Your question text here",
    "solution": "The solution here"
  },
  ...
]""")
        custom_filepath = input("\nEnter the path to your custom JSON file: ").strip()
        if not os.path.exists(custom_filepath):
            print(f"Warning: File {custom_filepath} does not exist.")
            custom_filepath = input("Please enter a valid path or press Enter to use default: ").strip()
            if not custom_filepath or not os.path.exists(custom_filepath):
                custom_filepath = "data/custom_questions.json"
                print(f"Using default path: {custom_filepath}")
                
                # Create the data directory if it doesn't exist
                os.makedirs(os.path.dirname(custom_filepath), exist_ok=True)
                
                # Check if the default file exists, if not create a simple example
                if not os.path.exists(custom_filepath):
                    print("Default file not found. Creating an example file...")
                    example_questions = [
                        {
                            "question": "What is 2+2?",
                            "solution": "4"
                        },
                        {
                            "question": "What is the square root of 16?",
                            "solution": "4"
                        }
                    ]
                    with open(custom_filepath, 'w', encoding='utf-8') as f:
                        json.dump(example_questions, f, indent=2)
                    print(f"Created example file at {custom_filepath}")
    
    # Get model to evaluate
    model = input("\nEnter the model to evaluate (e.g., gpt-4-turbo): ").strip()
    
    # Get evaluator model
    evaluator_model = input("\nEnter the evaluator model (default: claude-3-opus-20240229): ").strip() or "claude-3-opus-20240229"
    
    # Get number of questions
    while True:
        try:
            num_questions = int(input("\nEnter the number of questions to evaluate: "))
            if num_questions > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    return dataset_choice, model, evaluator_model, num_questions, custom_filepath

async def main():
    # Ensure logs directory exists
    os.makedirs(LOGS_FOLDER, exist_ok=True)
    
    # Get user input
    dataset_choice, model, evaluator_model, num_questions, custom_filepath = get_user_input()
    
    # Get dataset name for reporting using the abstracted function
    dataset_name = datasets.get_dataset_by_option(dataset_choice)
    
    print(f"\nFetching {num_questions} questions from {dataset_name}...")
    # Use the abstracted function to get questions, passing the custom filepath if needed
    question_pairs = datasets.get_questions_by_option(dataset_choice, num_questions, custom_filepath)
    
    if not question_pairs:
        print("Error: No questions could be loaded. Please check your dataset selection.")
        return
    
    print(f"Starting evaluation of {model} on {num_questions} questions...")
    
    # Process in batches
    all_results = []
    total_accurate = 0
    total_bad_responses = 0
    total_errors = 0
    total_time = 0

    for i in range(0, num_questions, BATCH_SIZE):
        batch = question_pairs[i:i+BATCH_SIZE]
        batch_results = await evaluate_batch(model, evaluator_model, batch)
        
        all_results.extend(batch_results['results'])
        total_accurate += batch_results['accurate_count']
        total_bad_responses += batch_results['bad_responses']
        total_errors += batch_results['errors']
        total_time += batch_results['time_taken']

        # Calculate and print progress
        processed = min(i+BATCH_SIZE, num_questions)
        current_accuracy = (total_accurate / processed) * 100 if processed > 0 else 0
        print(f"Processed {processed}/{num_questions} questions - Current Accuracy: {current_accuracy:.2f}%")

    # Calculate final metrics
    final_accuracy = (total_accurate / num_questions) * 100 if num_questions > 0 else 0
    
    # Calculate 95% confidence interval
    z_score = 1.96  # for 95% confidence
    margin_of_error = z_score * math.sqrt((final_accuracy * (100 - final_accuracy)) / num_questions)
    ci_lower = max(0, final_accuracy - margin_of_error)
    ci_upper = min(100, final_accuracy + margin_of_error)

    # Compile results
    results = {
        "dataset": dataset_name,
        "model": model,
        "evaluator_model": evaluator_model,
        "total_questions": num_questions,
        "accuracy": final_accuracy,
        "confidence_interval": (ci_lower, ci_upper),
        "bad_responses": total_bad_responses,
        "errors": total_errors,
        "time_taken": total_time,
        "detailed_results": all_results
    }
    
    # Print summary
    print("\n=== Evaluation Results ===")
    print(f"Dataset: {results['dataset']}")
    print(f"Model: {results['model']}")
    print(f"Evaluator Model: {results['evaluator_model']}")
    print(f"Total Questions: {results['total_questions']}")
    print(f"Accuracy: {results['accuracy']:.2f}% (95% CI: {results['confidence_interval'][0]:.2f}% - {results['confidence_interval'][1]:.2f}%)")
    print(f"Bad Responses: {results['bad_responses']}")
    print(f"Errors: {results['errors']}")
    print(f"Time Taken: {results['time_taken']:.2f} seconds")
    
    # Save results to file
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(LOGS_FOLDER, f"eval_{dataset_name}_{model.replace('/', '-')}_{num_questions}q_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())