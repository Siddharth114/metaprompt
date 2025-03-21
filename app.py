import streamlit as st
from config.settings import set_page_config
from components.sidebar import render_sidebar
from components.test_data_upload import render_test_data_upload
from components.results import display_optimization_results
from utils.optimizer import optimize_prompt
from utils.logging_util import setup_logger

def main():
    logger = setup_logger()
    
    set_page_config()
    
    st.title("Autonomous Reflective Prompt Engineering Agent")
    st.markdown(
        """
        Autonomous Reflective Prompt Engineering Agent that iteratively generates, tests, analyzes, and refines prompts to maximize task performance.
        """
    )
    
    params = render_sidebar()
    
    test_data = render_test_data_upload()
    
    if st.button("Start Optimization", disabled=(test_data is None)):
        with st.spinner("Optimizing..."):
            logger.info("Starting optimization process.")
            best_prompt, best_accuracy, results = optimize_prompt(
                test_data=test_data,
                max_iterations=params["max_iterations"],
                accuracy_threshold=params["accuracy_threshold"],
                reflection_temperature=params["reflection_temperature"],
            )
            
            display_optimization_results(best_prompt, best_accuracy, results)
            
            logger.info("Optimization process completed successfully.")

if __name__ == "__main__":
    main()