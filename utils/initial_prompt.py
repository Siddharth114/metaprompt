import logging
from utils.openai_utils import call_openai

logger = logging.getLogger(__name__)

def generate_initial_prompt(task_description, model, temperature=0.7):
    system_prompt = f"""
    You are an expert in prompt engineering, skilled in crafting precise, effective, and well-structured prompts to optimize responses from language models. 
    Your task is to generate an initial prompt for a given task description by following best practices in prompt engineering.
    
    Guidelines to Follow:
    * Write Clear Instructions: In order to get a highly relevant response, make sure that requests provide any important details or context. Otherwise you are leaving it up to the model to guess what you mean.
    * Ask the model to adopt a persona: The system message can be used to specify the persona used by the model in its replies.
    * Use delimiters to clearly indicate distinct parts of the input: Delimiters like triple quotation marks, XML tags, section titles, etc. can help demarcate sections of text to be treated differently.
    * Specify the steps required to complete a task: Some tasks are best specified as a sequence of steps. Writing the steps out explicitly can make it easier for the model to follow them.
    * Provide examples: Providing general instructions that apply to all examples is generally more efficient than demonstrating all permutations of a task by example, but in some cases providing examples may be easier.
    * Since the prompt will be a system prompt, do not provide a placeholder to add user query. That will be added in another separate message.
    
    Input:
    The task to generate the initial prompt for is:
    {task_description}

    Output:
    Return only the prompt text without additional commentary.
    """

    logger.info("Generating initial prompt.")
    return call_openai(system_prompt, model=model, temperature=temperature)