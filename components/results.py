import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

def display_optimization_results(best_prompt, best_accuracy, results_df):
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

    st.download_button(
        label="Download Results as CSV",
        data=results_df.to_csv(index=False),
        file_name=f"prompt_optimization_results_{datetime.now().isoformat()}.csv",
        mime="text/csv",
    )