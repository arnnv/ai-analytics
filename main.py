import time
from graph import app, GraphState
from utils import load_data
import plotly.io as pio
import streamlit as st

pio.templates.default = "plotly_white"

st.set_page_config(layout="wide")

st.title("📊 Autonomous Data Visualization Agent")
st.markdown("Upload a CSV file and let the AI generate insightful visualizations.")

# --- Sidebar for Upload and Control ---
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
max_iterations = st.sidebar.slider("Max Visualizations to Generate", 1, 10, 5)
analyze_button = st.sidebar.button("Analyze Data")

# --- Main Area for Displaying Results ---

if analyze_button and uploaded_file is not None:
    st.info(f"Starting analysis... Generating up to {max_iterations} charts.")
    start_time = time.time()

    df = load_data(uploaded_file)

    if df is not None:
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())

        initial_state: GraphState = {
            "dataframe": df,
            "profile_report": None,
            "visualization_ideas": [],
            "charts": [],
            "current_chart_idea": None,
            "current_chart_code": None,
            "current_chart_explanation": None,
            "current_chart_figure": None,
            "feedback": None,
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "error": None
        }

        final_state = None
        with st.spinner("Analyzing data and generating visualizations..."):
            try:
                final_state = app.invoke(initial_state)

            except Exception as e:
                st.error(f"An error occurred during the graph execution: {e}")
                if 'error' not in initial_state or not initial_state['error']:
                     initial_state['error'] = str(e)
                final_state = initial_state

        end_time = time.time()
        st.success(f"Analysis finished in {end_time - start_time:.2f} seconds.")

        # --- Display Results ---
        if final_state:
            if final_state.get("error"):
                st.warning(f"Process may have completed with errors: {final_state['error']}")

            charts = final_state.get('charts', [])
            if charts:
                st.header("Generated Visualizations")
                for i, chart_data in enumerate(charts):
                    st.subheader(f"Chart {i+1}: {chart_data.get('idea', 'Untitled')}")
                    col1, col2 = st.columns([1, 2]) # Ratio 1:2 for code/explanation vs chart
                    with col1:
                        with st.expander("Show Generated Code"): # Optional: Show generated code
                            st.code(chart_data.get('code', '# No code available'), language='python')
                    with col2:
                        st.markdown("**Explanation:**")
                        st.markdown(chart_data.get('explanation', "*No explanation generated.*"))
                        st.markdown("**Visualization:**")

                        if chart_data.get('image_bytes'): # Prioritize using st.image with bytes for better control and consistency
                            st.image(chart_data['image_bytes'], use_container_width=True) # Use use_container_width=True to fit the column
                        elif chart_data.get('figure'): # Fallback to st.pyplot if bytes are missing but figure exists
                            st.warning("Image bytes missing, attempting to display figure object directly (may differ).")
                            st.pyplot(chart_data['figure'], clear_figure=True, use_container_width=True)
                        else:
                            st.warning("No visualization figure or image bytes available.")

                    st.divider()
            else:
                st.warning("No visualizations were successfully generated.")

            if final_state.get("profile_report"): # Optionally display the final profile report
                with st.expander("View Full Data Profile"):
                    st.text(final_state["profile_report"])

    else:
        st.error("Could not load data from the uploaded file. Please check the file format.")

elif analyze_button:
    # Case where button is clicked but no file is uploaded
    st.warning("Please upload a CSV file first.")
else:
    # Initial state of the app before upload/analysis
    st.info("Upload a CSV file and click 'Analyze Data' to begin.")