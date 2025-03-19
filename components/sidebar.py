import streamlit as st

def render_sidebar():
    """
    Render sidebar with optimization parameters
    """
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
    
    return {
        "max_iterations": max_iterations,
        "accuracy_threshold": accuracy_threshold,
        "reflection_temperature": reflection_temperature,
    }