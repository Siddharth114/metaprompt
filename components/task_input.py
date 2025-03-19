import streamlit as st

def render_task_input():
    st.header("Task Definition")
    task_description = st.text_area(
        "Describe the task you want to optimize a prompt for:",
        height=100,
        placeholder="Example: Classify customer emails as either 'Complaint', 'Question', or 'Feedback'",
    )
    
    return task_description