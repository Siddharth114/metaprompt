import logging
from openai import OpenAI
import streamlit as st
from config.settings import DEFAULT_API_KEY, DEFAULT_MODEL

logger = logging.getLogger(__name__)

openai = OpenAI(api_key=DEFAULT_API_KEY)

def call_openai(prompt, model=DEFAULT_MODEL, temperature=0.0):
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