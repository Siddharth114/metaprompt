import json
import logging
import streamlit as st
import pandas as pd
import time
from utils.initial_prompt import generate_initial_prompt
from utils.evaluation import evaluate_prompt
from utils.openai_utils import call_openai


logger = logging.getLogger(__name__)


def reflect_and_improve(prompt, evaluation_results, accuracy, temperature):
    reflection_prompt = f"""
    You are an AI prompt optimizer designed to iteratively improve prompts based on testing results. Your goal is to analyze the effectiveness of the current prompt in achieving the given task, identify what is working and what is not based on the provided accuracy and testing results, and generate an improved prompt.

    Instructions:
    1. Understand the Task:
        * Identify the goal of the task.

    2. Analyze the Testing Results:
        * Examine the accuracy and testing results to determine the strengths and weaknesses of the current prompt.
        * Identify patterns of failure and success.

    3. Reflection on the Prompt:
        * What aspects of the prompt are contributing to correct outputs?
        * What parts of the prompt might be causing errors or inefficiencies?
        * Is the prompt too vague, too restrictive, or missing key instructions?
        * Is the output format consistent with what the expectation is?

    4. Generate an Updated Prompt:
        * Improve clarity, specificity, or structure to enhance performance.
        * Address identified issues while retaining effective components.
        * Ensure the prompt aligns with the task objective and optimizes accuracy.

    Inputs:

    Task: 
    
    The LLM receiving this system prompt will be responsible for generating answers to user queries based on retrieved contextual information from a vector database. 
    The goal is to answer the user’s question accurately using only the provided context and cite sources appropriately.
        
    Current Prompt:
    "{prompt}"

    Current Accuracy: {accuracy:.2f}%

    Results from testing:

    {json.dumps(evaluation_results, indent=2)}

    Format your response as follows:
        
    REFLECTION:
    [Your detailed analysis of the current prompt's strengths and weaknesses]

    IMPROVED PROMPT:
    [The complete new prompt text without any additional information]
    """

    logger.info("Generating reflection and improved prompt.")
    
    messages = [{"role": "system", "content": reflection_prompt}]
    response = call_openai(messages=messages, temperature=temperature)

    try:
        reflection_part = response.split("REFLECTION:")[1].split("IMPROVED PROMPT:")[0].strip()
        improved_prompt = response.split("IMPROVED PROMPT:")[1].strip()
        logger.info("Successfully parsed reflection and improved prompt.")
        return reflection_part, improved_prompt
    except Exception as e:
        logger.warning(f"Couldn't parse reflection response properly: {e}")
        st.warning("Couldn't parse reflection response properly. Using raw response.")
        return response, prompt

def optimize_prompt(test_data, max_iterations=5, accuracy_threshold=95, reflection_temperature=1.0):
    iterations = list(range(1, max_iterations + 1))
    accuracies = []
    prompts = []
    reflections = []
    all_evaluation_results = []
    
    average_response_times = []
    average_evaluation_times = []
    reflection_times = []

    with st.spinner("Generating initial prompt..."):
        start_time = time.time()
        current_prompt = generate_initial_prompt()
        end_time = time.time()
        st.info("Generated initial prompt")
        logger.info(f"Generated initial prompt in {end_time - start_time:.2f} seconds.")

    progress_bar = st.progress(0)
    iteration_header = st.empty()
    current_prompt_container = st.empty()
    accuracy_container = st.empty()
    evaluation_container = st.empty()
    reflection_container = st.empty()

    for i in range(max_iterations):
        iteration_header.header(f"Iteration {i+1}/{max_iterations}")
        current_prompt_container.text_area(
            "Current Prompt:",
            value=current_prompt,
            height=150,
            disabled=True,
            key=f"prompt_{i}",
        )

        with st.spinner(f"Evaluating prompt on test data... (Iteration {i+1})"):
            accuracy, evaluation_results, average_response_time, average_evaluation_time = evaluate_prompt(
                current_prompt, test_data
            )

        accuracies.append(accuracy)
        prompts.append(current_prompt)
        all_evaluation_results.append(evaluation_results)
        average_response_times.append(average_response_time)
        average_evaluation_times.append(average_evaluation_time)

        accuracy_container.metric("Accuracy", f"{accuracy:.2f}%")

        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_container.dataframe(evaluation_df.drop(columns=["response_time", "evaluation_time"]))

        if accuracy >= accuracy_threshold:
            st.success(
                f"🎉 Accuracy threshold of {accuracy_threshold}% reached! Optimization complete."
            )
            logger.info(
                f"Accuracy threshold of {accuracy_threshold}% reached. Optimization complete."
            )
            reflections.append(f"Accuracy threshold of {accuracy_threshold}% reached.")
            break

        with st.spinner(f"Generating reflection and improvements... (Iteration {i+1})"):
            start_time = time.time()
            reflection, improved_prompt = reflect_and_improve(
                current_prompt, evaluation_results, accuracy, reflection_temperature
            )
            end_time = time.time()
            reflection_time = end_time - start_time
            reflection_times.append(reflection_time)
            logger.info(f"Generated reflection and improved prompt in {reflection_time:.2f} seconds.")

        reflections.append(reflection)
        reflection_container.markdown(f"### Reflection\n{reflection}")

        current_prompt = improved_prompt

        progress_bar.progress((i + 1) / max_iterations)
    
    results_df = pd.DataFrame({
        "Iteration": iterations[:i+1],
        "Accuracy": accuracies[:i+1],
        "Prompt": prompts[:i+1],
        "Reflection": reflections[:i+1],
        "Average Response Time": average_response_times[:i+1],
        "Average Evaluation Time": average_evaluation_times[:i+1],
        "Reflection Times": reflection_times[:i+1]
    })
    
    return prompts[accuracies.index(max(accuracies))], max(accuracies), results_df