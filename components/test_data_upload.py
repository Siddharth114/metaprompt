import streamlit as st
import json
import logging

logger = logging.getLogger(__name__)

def render_test_data_upload():
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
        except Exception as e:
            st.error(f"Error loading test data: {e}")
            logger.error(f"Error loading test data: {e}")
    
    return test_data