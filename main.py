import streamlit as st
from openai import OpenAI
import pandas as pd
import json
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

openai = OpenAI(api_key="sk-proj-vP2FDWY8FqaOj7ZP3h1FT3BlbkFJQMfLASTkP5mnRZetoiaL")

st.set_page_config(
    page_title="Autonomous Reflective Prompt Engineering Agent", layout="wide"
)

st.title("Autonomous Reflective Prompt Engineering Agent")
st.markdown(
    """
Autonomous Reflective Prompt Engineering Agent that iteratively generates, tests, analyzes, and refines prompts to maximize task performance. Unlike naive prompt optimization, PromptForge ensures that effective instructions are preserved while only modifying weaker components, preventing regression in performance.
"""
)

selected_model = "gpt-4o-mini"

st.sidebar.header("Optimization Parameters")
max_iterations = st.sidebar.slider(
    "Maximum Iterations:", min_value=1, max_value=20, value=5
)
accuracy_threshold = st.sidebar.slider(
    "Accuracy Threshold to Stop (%):", min_value=50, max_value=100, value=95
)
reflection_temperature = st.sidebar.slider(
    "Reflection Temperature:", min_value=0.0, max_value=2.0, value=1.0, step=0.1
)

st.header("Task Definition")
task_description = st.text_area(
    "Describe the task you want to optimize a prompt for:",
    height=100,
    placeholder="Example: Classify customer emails as either 'Complaint', 'Question', or 'Feedback'",
)

starter_prompt = st.text_area(
    "Starter Prompt (Optional):",
    height=150,
    placeholder="Enter an initial prompt to start with, or leave blank for auto-generation",
)

st.header("Test Data")
st.markdown(
    "Upload a JSON file with test cases. Format should be a list of objects with 'input' and 'expected_output' fields."
)

test_data_file = st.file_uploader("Upload Test Data (JSON)", type=["json"])
example_test_data = """
[
    {
        "input": "I'm very unhappy with your service, it's terrible!",
        "expected_output": "Complaint"
    },
    {
        "input": "When will my order arrive?",
        "expected_output": "Question"
    }
]
"""
test_data = None
if test_data_file is None:
    st.code(example_test_data, language="json")
else:
    try:
        test_data = json.load(test_data_file)
        st.success(f"Loaded {len(test_data)} test cases")
        with st.expander("View Test Data"):
            st.write(test_data)
        logger.info(
            f"Successfully loaded {len(test_data)} test cases from uploaded file."
        )
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        logger.error(f"Error loading test data: {e}")


def call_openai(prompt, model=selected_model, temperature=0.0):
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        logger.error(f"OpenAI API Error: {e}")
        return None


def generate_initial_prompt(task_description):
    system_prompt = f"""
    I need a high-quality initial prompt for the following task:
    
    {task_description}
    
    Create a concise, clear, and effective prompt that would help an LLM perform this task well.
    Return only the prompt text without additional commentary.
    """

    logger.info("Generating initial prompt.")
    return call_openai(system_prompt, temperature=0.7)


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

        results.append(
            {
                "input": input_text,
                "expected": expected,
                "actual": actual,
                "correct": is_correct,
            }
        )

    accuracy = (correct / len(test_data)) * 100
    logger.info(f"Evaluation completed. Accuracy: {accuracy:.2f}%")
    return accuracy, results


def reflect_and_improve(prompt, evaluation_results, task_description, accuracy):
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

Task: {task_description}
    
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
    response = call_openai(reflection_prompt, temperature=reflection_temperature)

    try:
        reflection_part = (
            response.split("REFLECTION:")[1].split("IMPROVED PROMPT:")[0].strip()
        )
        improved_prompt = response.split("IMPROVED PROMPT:")[1].strip()
        logger.info("Successfully parsed reflection and improved prompt.")
        return reflection_part, improved_prompt
    except Exception as e:
        logger.warning(f"Couldn't parse reflection response properly: {e}")
        st.warning("Couldn't parse reflection response properly. Using raw response.")
        return response, prompt


def optimize_prompt(
    task_description,
    test_data,
    starter_prompt=None,
    max_iterations=5,
    accuracy_threshold=95,
):
    iterations = list(range(1, max_iterations + 1))
    accuracies = []
    prompts = []
    reflections = []
    all_evaluation_results = []

    current_prompt = starter_prompt
    if not current_prompt:
        with st.spinner("Generating initial prompt..."):
            current_prompt = generate_initial_prompt(task_description)
            st.info("Generated initial prompt")
            logger.info("Generated initial prompt.")

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
            accuracy, evaluation_results = evaluate_prompt(
                current_prompt, test_data, selected_model
            )

        accuracies.append(accuracy)
        prompts.append(current_prompt)
        all_evaluation_results.append(evaluation_results)

        accuracy_container.metric("Accuracy", f"{accuracy:.2f}%")

        evaluation_df = pd.DataFrame(evaluation_results)
        evaluation_container.dataframe(evaluation_df)

        if accuracy >= accuracy_threshold:
            st.success(
                f"🎉 Accuracy threshold of {accuracy_threshold}% reached! Optimization complete."
            )
            logger.info(
                f"Accuracy threshold of {accuracy_threshold}% reached. Optimization complete."
            )
            break

        with st.spinner(f"Generating reflection and improvements... (Iteration {i+1})"):
            reflection, improved_prompt = reflect_and_improve(
                current_prompt, evaluation_results, task_description, accuracy
            )

        reflections.append(reflection)
        reflection_container.markdown(f"### Reflection\n{reflection}")

        current_prompt = improved_prompt

        progress_bar.progress((i + 1) / max_iterations)

    st.header("Optimization Results")

    results_df = pd.DataFrame(
        {
            "Iteration": iterations,
            "Accuracy": accuracies,
            "Prompt": prompts,
            "Reflection": reflections,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, accuracies, marker="o", linestyle="-", color="blue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Prompt Optimization Progress")
    ax.grid(True)
    st.pyplot(fig)

    best_idx = accuracies.index(max(accuracies))
    st.header("Best Prompt")
    st.info(f"Best accuracy: {accuracies[best_idx]:.2f}% (Iteration {best_idx+1})")
    st.text_area(
        "Best Performing Prompt:",
        value=prompts[best_idx],
        height=200,
        key="best_prompt",
    )

    st.download_button(
        label="Download Results as CSV",
        data=results_df.to_csv(index=False),
        file_name="prompt_optimization_results.csv",
        mime="text/csv",
    )

    logger.info("Optimization process completed.")
    return prompts[best_idx], accuracies[best_idx], results_df


if st.button(
    "Start Optimization", disabled=(not task_description or test_data is None)
):
    with st.spinner("Starting optimization process..."):
        logger.info("Starting optimization process.")
        best_prompt, best_accuracy, results = optimize_prompt(
            task_description=task_description,
            test_data=test_data,
            starter_prompt=starter_prompt,
            max_iterations=max_iterations,
            accuracy_threshold=accuracy_threshold,
        )
        logger.info("Optimization process completed successfully.")
