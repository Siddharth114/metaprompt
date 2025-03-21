import logging
from openai import OpenAI
import streamlit as st
from config.settings import DEFAULT_API_KEY, DEFAULT_MODEL

logger = logging.getLogger(__name__)

openai = OpenAI(api_key=DEFAULT_API_KEY)

def call_openai(messages, temperature=0.0, json_format=False):
    response_format = {"type": "json_object"} if json_format else None
    try:
        response = openai.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        logger.error(f"OpenAI API Error: {e}")
        return None