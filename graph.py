import pandas as pd
from typing import List, Optional, Dict, Any
import seaborn as sns
import matplotlib.pyplot as plt
import traceback
import io
import base64
from pydantic import BaseModel, Field, ConfigDict
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from utils import get_basic_profiling # Assuming utils.py is in the same directory

# Load environment variables from .env file
load_dotenv()

# --- Graph State Definition ---
class GraphState(BaseModel):
    """Represents the state of our graph.

    Attributes:
        dataframe: The input pandas DataFrame.
        cleaned_dataframe: The DataFrame after cleaning.
        cleaning_summary: A list of strings summarizing the cleaning actions.
        profile_report: A string containing the basic data profile.
        analysis_history: List of dictionaries storing past analysis summaries.
        charts: List of dictionaries, where each dict contains {'code': str, 'explanation': str, 'figure': Matplotlib Figure}.
        next_chart_idea: The specific visualization idea planned for the next iteration.
        current_chart_code: Python code generated for the current visualization.
        current_chart_explanation: Markdown explanation for the current visualization.
        current_chart_figure: The actual Matplotlib figure object.
        current_chart_image_bytes: The PNG bytes of the figure.
        feedback: Feedback from the LLM on the quality of the last generated chart.
        iteration_count: Tracks how many visualizations have been attempted.
        max_iterations: Maximum number of visualizations to generate.
        error: Stores any error messages encountered during processing.
    """
    # Allow arbitrary types like pandas DataFrame and Matplotlib Figure
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: Optional[pd.DataFrame] = None
    cleaned_dataframe: Optional[pd.DataFrame] = None
    cleaning_summary: List[str] = Field(default_factory=list)
    profile_report: Optional[str] = None
    analysis_history: List[Dict[str, str]] = Field(default_factory=list) # Added
    charts: List[Dict[str, Any]] = Field(default_factory=list) # Store {'code': ..., 'explanation': ..., 'figure': ...}
    next_chart_idea: Optional[str] = None # Added rename
    current_chart_code: Optional[str] = None
    current_chart_explanation: Optional[str] = None
    current_chart_figure: Optional[Any] = None # Changed Optional[object] to Optional[Any] for better compatibility
    current_chart_image_bytes: Optional[bytes] = None # To store the PNG bytes of the figure
    feedback: Optional[str] = None
    iteration_count: int = 0
    max_iterations: int = 5 # Limit the number of charts to generate
    error: Optional[str] = None


# --- Initialize LLM ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
# llm = ChatOpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     model="google/gemini-2.0-flash-exp:free",
# )

print("LLM Initialized successfully.")

# --- Graph Nodes ---

def profile_data(state: GraphState) -> dict:
    """Generates a profile report for the cleaned data using pandas profiling."""
    print("---NODE: Profiling Data---")
    df = state.cleaned_dataframe # NEW: Use cleaned data
    if df is None:
        print("Error: No cleaned DataFrame available for profiling.") # NEW
        return {"error": "Cleaned DataFrame missing for profiling.", "profile_report": None} # NEW

    try:
        # Use pandas describe as a simple profile
        profile = get_basic_profiling(df)
        print("Profiling Report Generated.")
        # print(profile) # Optional: print profile for debugging
        return {"profile_report": profile, "error": None}
    except Exception as e:
        print(f"Error during profiling: {e}")
        return {"error": f"Profiling failed: {e}"}

def plan_visualizations(state: GraphState) -> dict:
    """Recommends the next single visualization based on profile and history."""
    print("---NODE: Planning Next Visualization---")
    if not llm:
        return {"error": "LLM not initialized.", "next_chart_idea": None}

    profile_report = state.profile_report # Profile now comes from cleaned data
    history = state.analysis_history
    cleaned_df = state.cleaned_dataframe # Get cleaned df for column list

    if cleaned_df is None:
         print("Error: Cleaned DataFrame missing for planning.")
         return {"error": "Cleaned DataFrame is missing for planning.", "next_chart_idea": None}
    if not profile_report:
        print("Warning: Profile report is missing, planning might be less accurate.")
        # Allow planning without profile, but it might be less ideal

    # Prepare history string
    history_str = "\n".join([f"- {item['idea']}" for item in history]) if history else "None yet."

    prompt = f"""You are a data visualization expert. Your task is to recommend the *single most insightful* next data visualization based on the provided data profile and the history of visualizations already generated. Focus on uncovering new patterns or relationships.

    Data Profile Summary (Post-Cleaning):
    {profile_report[:2000]} # Provide a snippet for context

    Columns available (Names only): {list(cleaned_df.columns)} # Use columns from cleaned_df

    Visualization History:
    {history_str}

    Instructions:
    1.  **Analyze the profile and history.**
    2.  **Suggest ONE clear, actionable visualization idea (e.g., 'scatter plot of Sales vs Profit colored by Region', 'histogram of Customer Age').**
    3.  **If no more insightful visualizations seem possible, respond with only the word 'None'.**
    4.  **Do not include the python code, just the idea string.**

    Respond with *only* the visualization idea string, or "None".

    Recommended Visualization Idea:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        idea = response.content.strip()
        print(f"LLM recommended idea: {idea}")

        if idea.lower() == "none":
             print("LLM indicated no further visualizations are needed.")
             return {"next_chart_idea": None, "error": None} # Signal to stop

        # Basic validation (ensure it's not empty)
        if not idea:
            print("Warning: LLM returned an empty idea.")
            # Decide how to handle - maybe retry or stop? Let's stop for now.
            return {"next_chart_idea": None, "error": "LLM returned empty idea.", }

        return {"next_chart_idea": idea, "error": None}

    except Exception as e:
        print(f"Error during visualization planning: {e}")
        error_trace = traceback.format_exc()
        print(error_trace)
        return {"error": f"LLM planning failed: {e}", "next_chart_idea": None}

def generate_visualization_code(state: GraphState) -> dict:
    """Generates Python code (Pandas/Seaborn/Matplotlib) for the selected visualization idea."""
    print("---NODE: Generating Visualization Code---")
    if not llm:
        return {"error": "LLM not initialized.", "current_chart_code": None}

    idea = state.next_chart_idea
    profile_report = state.profile_report # Profile from cleaned data
    # df = state.dataframe # OLD
    df = state.cleaned_dataframe # NEW: Use cleaned data

    if not idea:
        print("Error: No current visualization idea selected.")
        return {"error": "No visualization idea to generate code for.", "current_chart_code": None}
    if df is None:
        # print("Error: DataFrame is missing.") # OLD message
        print("Error: Cleaned DataFrame is missing.") # NEW message
        # return {"error": "DataFrame is missing, cannot generate code."} # OLD error
        return {"error": "Cleaned DataFrame is missing, cannot generate code.", "current_chart_code": None} # NEW error
    if not profile_report:
        print("Warning: Profile report is missing, code generation might be less accurate.")
        # Proceeding without profile might be okay, but less ideal

    # Construct the prompt for the LLM
    prompt = f"""You are a data visualization assistant. Given a dataset profile (from cleaned data) and a specific visualization idea, generate concise Python code using Pandas, Seaborn, and Matplotlib to create that plot.

    Dataset Profile Snippet (Post-Cleaning):
    {profile_report[:1500]} # Provide a snippet for context

    Columns available (Names only): {list(df.columns)} # df here is now cleaned_df

    Visualization Idea: {idea}

    Instructions:
    1.  **Import necessary libraries:** Assume `pandas as pd`, `seaborn as sns`, and `matplotlib.pyplot as plt` are available. The input DataFrame is named `df` (this df refers to the cleaned data).
    2.  **Data Preprocessing (If Needed):** Analyze the 'Visualization Idea'. If it requires data aggregation (e.g., calculating means, counts, sums per category), filtering, or reshaping, perform these steps first using pandas on the `df` DataFrame. **Store the result in a temporary DataFrame (e.g., `processed_df = df.groupby(...)`)**. Use this `processed_df` for plotting. Do *not* modify the original `df` directly unless that's part of creating the temporary frame. If no preprocessing is needed, use `df` directly for plotting.
    3.  **Set Figure Size:** Create a figure with a suitable size *before* plotting. Use `fig, ax = plt.subplots(figsize=(12, 6))` or `plt.figure(figsize=(12, 6))`.
    4.  **Generate Plot Code:** Write Seaborn/Matplotlib code to generate the plot described in the '{idea}', using the (potentially preprocessed) DataFrame.
    5.  **Use `ax`:** If you used `fig, ax = plt.subplots(...)`, make sure subsequent plotting calls use the `ax` object (e.g., `sns.histplot(..., ax=ax)`).
    6.  **Handle Potential Errors:** If the idea seems impossible with the data (e.g., requires non-existent columns, invalid preprocessing), return only the comment `# VisualizationError: Cannot generate plot for idea.`
    7.  **Rotate Labels (If Needed):** For plots with potentially many categorical labels (like bar charts or boxplots on x-axis), add `plt.xticks(rotation=45, ha='right')` to prevent overlap.
    8.  **Tight Layout:** End the code with `plt.tight_layout()` to adjust spacing.
    9.  **Conciseness:** Provide *only* the Python code block. Do not include explanations, comments (unless it's the error comment), or markdown formatting.

    Python Code:
    """

    try:
        # Use LLM to generate code
        print("Sending code generation request to LLM...")
        response = llm.invoke([HumanMessage(content=prompt)])
        code = response.content.strip()
        print(f"LLM Response for Code Gen:\n{code}")

        # Basic validation/cleaning of the response
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        if code.endswith("```"):
            code = code[:-len("```")].strip()

        # Check for error placeholder
        if "# VisualizationError:" in code:
            print(f"LLM indicated an error for idea '{idea}': {code}")
            # Optionally pass the error message back in the state
            error_msg = code.split(":", 1)[1].strip() if ":" in code else "LLM could not generate plot."
            return {"current_chart_code": None, "error": error_msg}

        # Check if code is empty after stripping
        if not code:
            print(f"Error: LLM returned empty code for idea '{idea}'.")
            return {"current_chart_code": None, "error": "LLM returned empty code."}

        # Store the generated code in the state
        print(f"Generated code for idea: {idea}")
        return {"current_chart_code": code, "error": None} # Return only modified fields

    except Exception as e:
        print(f"Error during code generation for '{idea}': {e}")
        error_trace = traceback.format_exc()
        print(error_trace)
        # Store the error in the state
        return {"current_chart_code": None, "error": f"LLM code generation failed: {e}"} # Return only modified fields

def execute_visualization_code(state: GraphState) -> dict:
    """Executes the generated Python code to create the visualization."""
    print("---NODE: Executing Visualization Code---")
    code = state.current_chart_code
    df = state.cleaned_dataframe # NEW: Use cleaned dataframe
    idea = state.next_chart_idea # Needed for error messages

    if not code:
        # This might happen if the previous node decided it couldn't generate code
        print("Error: No code found to execute.")
        return {"error": "No code provided for execution", "current_chart_figure": None, "current_chart_image_bytes": None}

    if df is None:
        # print("Error: DataFrame is missing for execution.") # OLD
        print("Error: Cleaned DataFrame is missing for execution.") # NEW
        # return {"error": f"DataFrame is missing, cannot execute code for '{idea}'"} # OLD
        return {"error": f"Cleaned DataFrame is missing, cannot execute code for '{idea}'", "current_chart_figure": None, "current_chart_image_bytes": None} # NEW

    if code.strip().startswith("# VisualizationError:"):
        error_msg = code.strip()
        print(f"Execution skipped due to generation error: {error_msg}")
        return {"error": f"Code generation failed: {error_msg}", "current_chart_figure": None, "current_chart_image_bytes": None}

    buffer = io.BytesIO()

    # Prepare execution context
    local_vars = {'df': df, 'pd': pd, 'sns': sns, 'plt': plt, 'np': np, 'fig': None, 'ax': None} # Pass cleaned 'df'
    global_vars = {}

    try:
        exec(code, global_vars, local_vars)

        # Try to get the figure generated by Seaborn/Matplotlib
        figure = None
        ax = local_vars.get('ax') # Get the object assigned to 'ax'
        
        if isinstance(ax, sns.PairGrid): # Check if it's a PairGrid
            print("Detected PairGrid object.")
            figure = ax.fig 
        elif hasattr(ax, 'get_figure'): # Check if it has get_figure (like AxesSubplot)
             print("Detected Axes object with get_figure.")
             figure = ax.get_figure()
        
        # Fallback: If 'ax' wasn't helpful or wasn't assigned, get the current figure
        if figure is None and plt.get_fignums():
            print("Falling back to plt.gcf().")
            figure = plt.gcf()

        img_bytes = None # Initialize image bytes

        if figure:
            print("Code executed successfully, figure generated.")
            plt.tight_layout() # Adjust layout before saving
            # Attempt to convert figure to image bytes using Matplotlib
            try:
                print("Attempting to save Matplotlib figure to PNG bytes...")
                figure.savefig(buffer, format='png', bbox_inches='tight', dpi=200) # Set DPI
                buffer.seek(0)
                img_bytes = buffer.read()
                buffer.close()
                plt.close(figure) # Close the figure to free memory
                print("Figure successfully converted to PNG bytes.")
            except Exception as e:
                print(f"Error converting Matplotlib figure to bytes: {e}")
                print(traceback.format_exc())
                # Close the figure even if saving fails
                plt.close(figure)
                return {"current_chart_figure": figure, "current_chart_image_bytes": None, "error": f"Error saving figure for '{idea}': {e}"} # Return only modified fields

            # Clear execution-specific errors if successful
            return {"current_chart_figure": figure, "current_chart_image_bytes": img_bytes, "error": None} # Return only modified fields
        else:
            print("Error: Code executed but did not produce a 'fig' variable or accessible Matplotlib figure.")
            plt.close('all') # Close any dangling figures
            return {"error": "Visualization code did not produce a 'fig' object or accessible Matplotlib figure.", "current_chart_figure": None, "current_chart_image_bytes": None} # Return only modified fields

    except Exception as e:
        print(f"Error executing visualization code: {e}")
        print(traceback.format_exc())
        # Ensure figure is closed even on error if it exists
        if 'fig' in local_vars and isinstance(local_vars['fig'], plt.Figure):
            plt.close(local_vars['fig'])
        elif plt.get_fignums(): # Try closing current figure if one exists
             plt.close(plt.gcf())
        return {"error": f"Error executing code for '{idea}': {e}", "current_chart_figure": None, "current_chart_image_bytes": None} # Return only modified fields

def generate_explanation(state: GraphState) -> dict:
    """Generates an explanation for the current visualization, including reasoning and interpretation."""
    print("---NODE: Generating Explanation---")
    idea = state.next_chart_idea
    profile_report = state.profile_report
    history = state.analysis_history
    img_bytes = state.current_chart_image_bytes
    code = state.current_chart_code # Get the generated code
    cleaning_summary = state.cleaning_summary # NEW: Fetch cleaning summary

    if not idea:
        print("Error: No visualization idea to explain.")
        # If there's no idea, there's likely no chart/explanation needed
        return {"error": "Missing visualization idea for explanation.", "current_chart_explanation": None}
    if not profile_report:
        print("Warning: Missing profile report for explanation.")
        # Allow explanation without profile, but it might be less insightful

    # Format history for the prompt
    history_str = "\n".join([f"- {item['idea']}: {item['explanation'][:100]}..." for item in history]) if history else "None yet."

    # Format cleaning summary for the prompt
    cleaning_str = "\n".join([f"- {action}" for action in cleaning_summary]) if cleaning_summary else "No specific cleaning actions were recorded."

    # Construct the main text prompt
    prompt_content = f"""You are a data analyst explaining a visualization and the steps taken to create it.

    Initial Data Cleaning Actions Performed:
    {cleaning_str} # NEW: Added cleaning summary

    Data Profile Summary (Post-Cleaning):
    {profile_report[:1000]}... # Include a snippet for context

    Current Visualization Idea: '{idea}'

    Visualization Code Used:
    ```python
    {code if code else '# No code was generated/executed for this step.'}
    ```

    History of Previous Visualizations:
    {history_str}

    Task:
    1.  **Reasoning:** Explain *why* this specific visualization ('{idea}') was likely chosen as the next step, considering the data profile (post-cleaning), the initial cleaning actions, and the visualizations generated previously ({history_str}). What new insight or unexplored aspect does it aim to address?
    2.  **Preprocessing Steps (within Code):** Look at the 'Visualization Code Used' above. Describe any data preprocessing steps (like filtering, grouping, aggregation using pandas) that were performed *within that code block* before the actual plotting command(s). If no preprocessing was done in the code block, state that clearly. (Do not repeat the initial cleaning actions here unless relevant to the specific code).
    3.  **Interpretation:** Briefly describe the final visualization created by the code. What does it show? What are the key takeaways or potential insights revealed by this specific chart?
    4.  **Format:** Provide the explanation in clear, concise Markdown. Structure it with headings for Reasoning, Preprocessing Steps (within Code), and Interpretation.

    Optional Input: An image of the visualization is also provided. Refer to it if helpful for the interpretation.
    """

    prompt_messages = [HumanMessage(content=prompt_content)]

    # Add image if available and LLM supports it
    if img_bytes and isinstance(llm, (ChatOpenAI, ChatGoogleGenerativeAI)):
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            }
        # Insert the image message after the initial text content
        # Adjusting structure for Gemini or OpenAI multimodal input
        if isinstance(llm, ChatGoogleGenerativeAI): # Gemini format
             prompt_messages[0].content = [prompt_messages[0].content, image_message]
        elif isinstance(llm, ChatOpenAI): # OpenAI format assumes list content
             if isinstance(prompt_messages[0].content, str):
                 prompt_messages[0].content = [ {"type": "text", "text": prompt_messages[0].content}, image_message ]
             else: # If already a list, append
                 prompt_messages[0].content.append(image_message)
        print("Image added to explanation prompt.")

    # Call the LLM
    try:
        print("Sending explanation request to LLM...")
        response = llm.invoke(prompt_messages)
        explanation = response.content.strip()
        print(f"LLM Explanation Received:\n{explanation[:200]}...") # Print start of explanation
        return {"current_chart_explanation": explanation, "error": None}
    except Exception as e:
        print(f"Error during explanation generation: {e}")
        error_trace = traceback.format_exc()
        print(error_trace)
        error_msg = f"LLM explanation failed: {e}"
        return {"current_chart_explanation": None, "error": error_msg}

def store_chart(state: GraphState) -> dict:
    """Stores the generated chart details and updates the analysis history."""
    print("---NODE: Storing Chart & Updating History---") # Updated print
    idea = state.next_chart_idea
    code = state.current_chart_code
    explanation = state.current_chart_explanation
    fig = state.current_chart_figure
    img_bytes = state.current_chart_image_bytes
    current_charts = state.charts
    history = state.analysis_history # Get current history

    if idea and fig is not None and explanation: # Check if we have the essentials
        chart_data = {
            "idea": idea,
            "code": code,
            "explanation": explanation,
            "figure": fig, # Store the figure object
            "image_bytes": img_bytes # Store the image bytes
        }
        updated_charts = current_charts + [chart_data]

        # Add to analysis history
        history_entry = {"idea": idea, "explanation": explanation}
        updated_history = history + [history_entry] # Append new analysis

        print(f"Stored chart and added to history for idea: '{idea}'")
        # Return only modified fields
        return {
            "charts": updated_charts,
            "analysis_history": updated_history, # Return updated history
             # Reset current chart fields for the next potential iteration
            "current_chart_code": None,
            "current_chart_explanation": None,
            "current_chart_figure": None,
            "current_chart_image_bytes": None,
            "next_chart_idea": None, # Clear the idea that was just completed
            "iteration_count": state.iteration_count + 1 # Increment iteration count here
            }
    else:
        print(f"Skipping storage/history update for idea '{idea}': Missing essential components (figure, explanation).")
        # If storage fails, don't update history, but still increment iteration count and clear fields
        return {
            "current_chart_code": None,
            "current_chart_explanation": None,
            "current_chart_figure": None,
            "current_chart_image_bytes": None,
            "next_chart_idea": None, # Clear the idea even if failed
            "iteration_count": state.iteration_count + 1 # Increment count even on failure
            } # Return empty dict if no state fields were modified except count/clear

def clean_data(state: GraphState) -> dict:
    """Cleans the input DataFrame using default strategies."""
    print("---NODE: Cleaning Data---")
    original_df = state.dataframe
    if original_df is None:
        print("Error: No DataFrame to clean.")
        return {"error": "Input DataFrame is missing for cleaning.", "cleaned_dataframe": None, "cleaning_summary": []}

    df = original_df.copy()
    summary = []
    initial_shape = df.shape

    # 1. Handle Missing Values
    missing_threshold = 0.5 # Drop columns with more than 50% missing
    cols_before = df.shape[1]
    df.dropna(axis=1, thresh=int(df.shape[0] * (1 - missing_threshold)), inplace=True)
    cols_after = df.shape[1]
    if cols_before > cols_after:
        summary.append(f"Dropped {cols_before - cols_after} columns with >{missing_threshold*100}% missing values.")

    rows_before = df.shape[0]
    # Impute numerical with median
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            summary.append(f"Imputed missing values in numerical column '{col}' with median ({median_val:.2f}).")

    # Impute categorical/object with mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            # Check if mode calculation is possible (e.g., not all NaN)
            if not df[col].isnull().all():
                try:
                    mode_val = df[col].mode()[0] # Take the first mode if multiple
                    df[col].fillna(mode_val, inplace=True)
                    summary.append(f"Imputed missing values in categorical column '{col}' with mode ('{mode_val}').")
                except IndexError:
                    summary.append(f"Skipped mode imputation for column '{col}' (could not compute mode, possibly all unique NaNs?).")
            else:
                 summary.append(f"Skipped imputation for column '{col}' as all values were missing.")

    # Optional: Drop rows with any remaining NaNs (should be few/none after imputation)
    df.dropna(axis=0, inplace=True)
    rows_after = df.shape[0]
    if rows_before > rows_after:
        summary.append(f"Dropped {rows_before - rows_after} rows containing remaining missing values.")

    # 2. Remove Duplicates
    duplicates_before = df.duplicated().sum()
    if duplicates_before > 0:
        df.drop_duplicates(inplace=True)
        summary.append(f"Removed {duplicates_before} duplicate rows.")

    # 3. Fix Incorrect Data Types (Basic Example: Try converting object columns that look numeric)
    for col in df.select_dtypes(include=['object']).columns:
        try:
            # Attempt conversion to numeric, coerce errors to NaN then check if any non-NaN exist
            converted = pd.to_numeric(df[col], errors='coerce')
            if not converted.isnull().all(): # If at least one value converted successfully
                 # Check if conversion to int is possible without data loss
                 if not converted.isnull().any() and (converted == converted.astype(int)).all(): # Check if all non-null values are integers
                     df[col] = converted.astype(int) # Convert to int if possible
                     summary.append(f"Converted column '{col}' type to integer.")
                 else:
                     df[col] = converted # Keep as float (original conversion) or potentially int64 containing NaN
                     summary.append(f"Converted column '{col}' type to numeric (float or int with NaN).")
        except Exception:
            pass # Ignore columns that can't be converted

    # 4. Correct Inconsistent Formatting (Very basic example: Trim whitespace from strings)
    for col in df.select_dtypes(include=['object']).columns:
        # Check if the column actually contains string data before attempting string operations
        is_string_col = df[col].apply(type).eq(str).any()
        if is_string_col:
            original_col_str = df[col].astype(str)
            stripped_col_str = original_col_str.str.strip()
            if not stripped_col_str.equals(original_col_str):
                df[col] = stripped_col_str
                summary.append(f"Trimmed leading/trailing whitespace from string values in column '{col}'.")


    final_shape = df.shape
    summary.insert(0, f"Data cleaning started with shape {initial_shape}, finished with shape {final_shape}.")
    print("Cleaning Summary:")
    for item in summary:
        print(f"- {item}")

    return {"cleaned_dataframe": df, "cleaning_summary": summary, "error": None}

# --- Conditional Edge Logic ---

def check_plan_result(state: GraphState) -> str:
    """Determines whether a visualization idea was planned."""
    print("---DECISION: Check Plan Result---")
    # Check for critical planning error first
    if state.error and ("planning failed" in state.error.lower() or "profile report" in state.error.lower()):
         print(f"Ending: Critical error before/during planning ({state.error}).")
         return "end_workflow" # Critical failure, end immediately

    # Check if an idea was successfully generated
    if state.next_chart_idea:
        print(f"Plan Result: Proceeding with idea '{state.next_chart_idea}'.")
        # Reset error if it wasn't a critical planning error
        # state.error = None # Maybe reset non-critical errors here? Let's skip for now.
        return "generate_visualization" # Route to generate code
    else:
        # This happens if LLM returned "None" or a non-critical error occurred during planning
        print("Plan Result: No visualization idea planned or planning indicated completion. Ending workflow.")
        return "end_workflow" # Route to end node

def check_iterations(state: GraphState) -> str:
    """Determines whether to continue to the next planning iteration."""
    print("---DECISION: Check Iterations---")
    iteration = state.iteration_count
    max_iterations = state.max_iterations

    # Optional: Check for error during the last chart generation/execution/explanation
    if state.error and state.error is not None : # Check if an error exists from the last cycle
        print(f"Warning: Error encountered in last cycle: {state.error}. Continuing if iterations allow.")
        # We proceed even if the last chart failed, as long as iterations < max

    if iteration >= max_iterations:
        print(f"Ending: Reached max iterations ({iteration}/{max_iterations}).")
        return "end_workflow"
    else:
        print(f"Continuing: Iteration {iteration+1}/{max_iterations} planned next.")
        return "plan_next_visualization" # Route back to planning

# --- Build the Graph ---

workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("clean_data", clean_data) # New node
workflow.add_node("profile_data", profile_data)
workflow.add_node("plan_visualizations", plan_visualizations)
# Removed: select_next_visualization node
workflow.add_node("generate_visualization_code", generate_visualization_code)
workflow.add_node("execute_visualization_code", execute_visualization_code)
workflow.add_node("generate_explanation", generate_explanation)
workflow.add_node("store_chart", store_chart)
workflow.add_node("end_node", lambda state: print(f"---Workflow Finished---\nFinal State:\nCharts: {len(state.charts)}\nHistory: {len(state.analysis_history)}\nIterations: {state.iteration_count}\nError: {state.error}")) # More informative end node

# Define edges
workflow.set_entry_point("clean_data") # Start with cleaning
# workflow.set_entry_point("profile_data") # OLD entry point
workflow.add_edge("clean_data", "profile_data") # Clean -> Profile
workflow.add_edge("profile_data", "plan_visualizations") # Profile -> Plan
# Conditional edge after planning
workflow.add_conditional_edges(
    "plan_visualizations",
    check_plan_result,
    {
        "generate_visualization": "generate_visualization_code",
        "end_workflow": "end_node"
    }
)

# Core visualization sequence
workflow.add_edge("generate_visualization_code", "execute_visualization_code")
workflow.add_edge("execute_visualization_code", "generate_explanation")
workflow.add_edge("generate_explanation", "store_chart")

# Conditional edge after storing (looping logic)
workflow.add_conditional_edges(
    "store_chart",
    check_iterations,
    {
        "plan_next_visualization": "plan_visualizations", # Loop back to plan the *next* step
        "end_workflow": "end_node" # Exit the graph
    }
)

# Define the final endpoint (redundant as conditional edges handle it, but safe to keep)
# workflow.add_edge("end_node", END) # This might be implicitly handled

# Compile the graph
# Increase recursion limit using with_config after compiling
app = workflow.compile().with_config({'recursion_limit': 50})

print("Graph Compiled Successfully.")
