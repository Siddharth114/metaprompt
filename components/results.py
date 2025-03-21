import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import zipfile
import io
import logging

logger = logging.getLogger(__name__)

def display_optimization_results(best_prompt, best_accuracy, results_df, all_evaluation_results):
    st.header("Optimization Results")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results_df["Iteration"],
            y=results_df["Accuracy"],
            mode="lines+markers",
            marker=dict(color="blue"),
            line=dict(color="blue"),
            hovertemplate="Iteration: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Prompt Optimization Progress",
        xaxis_title="Iteration",
        yaxis_title="Accuracy (%)",
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
        showlegend=False,
        template="plotly_white",
    )

    st.plotly_chart(fig)

    st.header("Best Prompt")
    st.info(f"Best accuracy: {best_accuracy:.2f}%")
    st.text_area(
        "Best Performing Prompt:",
        value=best_prompt,
        height=200,
        key="best_prompt",
    )

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.writestr("optimization_results.csv", results_df.to_csv(index=False))

        for i, eval_result in enumerate(all_evaluation_results):
            eval_result_df = pd.DataFrame.from_records(eval_result)

            if isinstance(eval_result_df, pd.DataFrame):
                zipf.writestr(f"evaluation_results_iteration_{i+1}.csv", eval_result_df.to_csv(index=False))
            else:
                logging.warning(f"Warning: Skipping non-DataFrame evaluation result at index {i+1}.")

    zip_buffer.seek(0)

    st.download_button(
        label="Download Results as ZIP",
        data=zip_buffer,
        file_name=f"prompt_optimization_results_{datetime.now().isoformat()}.zip",
        mime="application/zip",
    )