import logging
from utils.openai_utils import call_openai

logger = logging.getLogger(__name__)

def generate_initial_prompt(temperature=0.7):
    system_prompt = """
    You are an AI assistant that specializes in designing system prompts for LLMs. 
    Your task is to generate a first draft of a system prompt that instructs an LLM to perform the answering stage of a Retrieval-Augmented Generation (RAG) workflow.
    
    Task Overview:
    The LLM receiving this system prompt will be responsible for generating answers to user queries based on retrieved contextual information from a vector database. 
    The goal is to answer the user’s question accurately using only the provided context and cite sources appropriately.
    
    Requirements for the System Prompt:
    1. The LLM will recieve:
        * A user question
        * Retrieved contexts from a vector database related to the question
    2. The LLM’s response should:
        * Answer the user’s question based only on the retrieved contexts
        * Cite the relevant contexts when making factual claims (e.g., using inline references like [Source 1])
        * If the retrieved contexts do not contain enough information, clearly state that the answer is not available rather than making up information
        * Ensure coherent, well-structured, and professional responses
    3. The system prompt should be clear, concise, and actionable for an LLM to follow, written in a Chain-of-Thought (CoT) format.

    Expected Output:
    Your response should contain a first draft of the system prompt that meets these requirements.
    Avoid unnecessary explanations or meta-level commentary—focus on making the system prompt effective.
    """

    logger.info("Generating initial prompt.")
    messages = [{"role":"system", "content": system_prompt}]
    response, prompt_tokens, completion_tokens = call_openai(messages=messages, temperature=temperature)
    return response, prompt_tokens, completion_tokens