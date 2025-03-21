import json
import logging
import time

from utils.openai_utils import call_openai

logger = logging.getLogger(__name__)

def evaluate_answer(question, expected_answer, actual_answer):
    system_prompt= """You are an AI assistant responsible for evaluating a newly generated answer against a reference answer (old answer) for a specific question.
Your task is to analyze the new answer thoroughly and provide a fair assessment based on the provided criteria.
Ensure you fully understand the criteria before starting the evaluation. Use the specified format for your assessment without deviation
    """
    user_prompt = f"""Here's the question you'll be evaluating answers for:
                    <question>
                    {question}
                    </question>
                    The correct answer (expected answer) is:
                    <expected_answer>
                    {expected_answer}
                    </expected_answer>
                    Now, here's the new answer you need to evaluate:
                    <actual_answer>
                    {actual_answer}
                    </actual_answer>
                    Evaluate the new answer based on the following criteria:
                    1. Improvement
                      Case:
                      Old Answer: "I cannot answer due to lack of information" or similar phrases indicating no relevant information is provided.
                      New Answer: Provides relevant details or addresses the query.
                      Action: Mark as Improvement.
                      Can only consider if the old answer is "I cannot answer due to lack of information" or similar phrases indicating no relevant information is provided.
                      Exception:
                      -If both answers state: "I don't know," "I cannot answer due to lack of information," or similar phrases indicating no relevant information is provided
                      -If both answers are identical or both are incapable of addressing the query, do not mark as Improvement
                    2. Degraded
                      Case:
                      Old Answer: Provides relevant details.
                      New Answer: "I cannot answer due to lack of information" or similar phrases indicating no relevant information is provided.
                      Action: Mark as Degraded.
                      Can only consider if the new answer is "I cannot answer due to lack of information" or similar phrases indicating no relevant information is provided.
                      Exception:
                      -If both answers state: "I don't know," "I cannot answer due to lack of information," or similar phrases indicating no relevant information is provided
                      -If both answers are identical or both are incapable of addressing the query, do not mark as Degraded
                    3. Information Gain
                      Case:
                      Old Answer: Provides partial information.
                      New Answer: Adds more accurate, relevant, or detailed information.
                      Action: Mark as Information Gain.
                      Can only consider if the old answer and new answer have some relevant information
                      -If both answers are "I don't know," "I cannot answer due to lack of information," or similar phrases indicating no relevant information is provided
                      -If Old answer is "I don't know," "I cannot answer due to lack of information," or similar phrases indicating no relevant information is provided, do not mark as Information Gain.
                      4. Information Gap
                      Case:
                      Old Answer: Contains more relevant, accurate, or detailed information.
                      New Answer: Lacks detail or relevance compared to the old answer.
                      Action: Mark as Information Gap.
                      Can only consider if the old answer and new answer have some relevant information
                      Exception:
                      -If both answers are "I don't know," "I cannot answer due to lack of information," or similar phrases indicating no relevant information is provided
                      -If New answer is "I don't know," "I cannot answer due to lack of information," or similar phrases indicating no relevant information is provided, do not mark as Information Gap.
                    5. Similarity
                      Case:
                      Old Answer: Provides relevant information.
                      New Answer: Similar in quality, content, and relevance to the old answer.
                      Action: Mark as Similar.
                      Can only consider if the old answer and new answer have some relevant information
                      Exception:
                      -If the old answer contains more relevant information but the new answer is Inadequate of addressing the question
                      -If one of the answers is "I don't know," "I cannot answer due to lack of information," or similar phrases indicating no relevant information is provided, do not mark as Similar
                    6. Inadequate
                      Case:
                      Old Answer: "I don't know," "I cannot answer due to lack of information," or similar phrases indicating no relevant information is provided.
                      New Answer: "I don't know," "I cannot answer due to lack of information," or similar phrases indicating no relevant information is provided.
                      Action: Mark as Inadequate.
                      Can only consider if the old answer and new answer are "I don't know," "I cannot answer due to lack of information," or similar phrases indicating no relevant information is provided
                      Exception:
                      -Do not mark as Inadequate if the old answer contains any relevant information
                      -Do not mark as Inadequate if the new answer contains any relevant information
                      -Do not mark as Inadequate if answer is similar in quality, content, and relevance to the old answer
                    Please follow these steps for your evaluation:
                    1. Carefully read and understand the question, old answer, and new answer.
                    2. Compare the new answer to the old answer, considering the evaluation criteria.
                    3. Analyze the new answer in relation to the question and the old answer.
                    4. Determine if there has been an Improvement, Degraded, Inadequate, Information Gain, Information Gap or if the answers are Similar based on the evaluation criteria and stricly follow the criteria.
                    5. You can only choose one of the following actions: Improvement, Degraded, Inadequate, Information Gain, Information Gap or Similar
                    6. Provide your assessment and reasoning in the following format:
                    Here's an example of how your JSON output should look:
                    {{
                      "analysis": "Provide a detailed analysis of the new answer compared to the old answer, addressing each of the evaluation criteria. Explain your reasoning for your assessment",
                      "result": "State whether the new answer has `Improvement`,`Degraded`,'Inadequate` ,`Information Gain`, `Information Gap` or `Similar` to the old answer."
                    }}
                    Remember to be objective and thorough in your evaluation, considering all aspects of the answers in relation to the question and old answer."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},]
    
    response, prompt_tokens, completion_tokens = call_openai(messages=messages, json_format=True, temperature=1)
    
    response_json = json.loads(response)
    
    result = response_json["result"]
    
    is_correct = result in ['Improvement', 'Information Gain', 'Similar']
    
    return is_correct, prompt_tokens, completion_tokens
    

def evaluate_prompt(prompt, test_data):
    results = []
    correct = 0
    response_times = []
    evaluation_times = []
    all_response_prompt_tokens = []
    all_response_completion_tokens = []
    
    all_evaluation_prompt_tokens = []
    all_evaluation_completion_tokens = []
    
    logger.info(f"Evaluating prompt with {len(test_data)} test cases.")
    for test_case in test_data:
        input_text = test_case["input"]
        question = input_text['question']
        expected_answer = test_case["expected_output"]

        messages = [{"role": "system", "content": prompt}, {"role": "user", "content":json.dumps(input_text)}]

        start_time = time.time()
        actual_answer, response_prompt_tokens, response_completion_tokens = call_openai(messages=messages)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        response_times.append(response_time)
        all_response_prompt_tokens.append(response_prompt_tokens)
        all_response_completion_tokens.append(response_completion_tokens)

        start_time = time.time()
        is_correct, evaluation_prompt_tokens, evaluation_completion_tokens = evaluate_answer(question, expected_answer, actual_answer)
        end_time = time.time()
        
        evaluation_time = end_time - start_time
        
        evaluation_times.append(evaluation_time)
        all_evaluation_prompt_tokens.append(evaluation_prompt_tokens)
        all_evaluation_completion_tokens.append(evaluation_completion_tokens)
        
        
        if is_correct:
            correct += 1
        
        results.append({
            "question": question,
            "expected": expected_answer,
            "actual": actual_answer,
            "correct": is_correct,
            "response_time": response_time,
            "response_prompt_tokens": response_prompt_tokens,
            "response_completion_tokens": response_completion_tokens,
            "evaluation_time": evaluation_time,
            "evaluation_prompt_tokens": evaluation_prompt_tokens,
            "evaluation_completion_tokens": evaluation_completion_tokens
        })

    accuracy = (correct / len(test_data)) * 100
    logger.info("Evaluation completed.")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    
    average_response_time = round(sum(response_times) / len(response_times), 2)
    average_evaluation_time = round(sum(evaluation_times) / len(evaluation_times), 2)
    
    average_response_prompt_tokens = round(sum(all_response_prompt_tokens) / len(all_response_prompt_tokens), 2)
    average_response_completion_tokens = round(sum(all_response_completion_tokens) / len(all_response_completion_tokens), 2)
    
    average_evaluation_prompt_tokens = round(sum(all_evaluation_prompt_tokens) / len(all_evaluation_prompt_tokens))
    average_evaluation_completion_tokens = round(sum(all_evaluation_completion_tokens) / len(all_evaluation_completion_tokens))
    
    
    logger.info(f"Average response time per test case: {average_response_time} seconds")
    logger.info(f"Average evaluation time per test case: {average_evaluation_time} seconds")
    logger.info(f"Average response prompt tokens per test case: {average_response_prompt_tokens}")
    logger.info(f"Average response completion tokens per test case: {average_response_completion_tokens}")
    
    return accuracy, results, average_response_time, average_evaluation_time, average_response_prompt_tokens, average_response_completion_tokens, average_evaluation_prompt_tokens, average_evaluation_completion_tokens