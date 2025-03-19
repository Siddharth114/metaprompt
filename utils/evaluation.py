import logging
from utils.openai_utils import call_openai

logger = logging.getLogger(__name__)

def evaluate_prompt(prompt, test_data, model):
    results = []
    correct = 0
    
    logger.info(f"Evaluating prompt with {len(test_data)} test cases.")
    for i, test_case in enumerate(test_data):
        input_text = test_case["input"]
        expected = test_case["expected_output"]

        full_prompt = f"{prompt}\n\nInput: {input_text}"

        actual = call_openai(full_prompt, model=model)

        is_correct = expected.strip().lower() == actual.strip().lower()
        if is_correct:
            correct += 1

        results.append({
            "input": input_text,
            "expected": expected,
            "actual": actual,
            "correct": is_correct,
        })

    accuracy = (correct / len(test_data)) * 100
    logger.info(f"Evaluation completed. Accuracy: {accuracy:.2f}%")
    return accuracy, results