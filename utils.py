import pandas as pd
import io
from typing import Optional, Any

def load_data(uploaded_file: Optional[Any]) -> Optional[pd.DataFrame]:
    """Loads data from an uploaded CSV file into a pandas DataFrame."""
    if uploaded_file is not None:
        try:
            # Use io.StringIO to read the file content
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    return None


def get_basic_profiling(df: pd.DataFrame) -> str:
    """Generates a basic profile of the DataFrame."""
    if df is None:
        return "No data loaded."

    profile = {
        "Shape": df.shape,
        "Columns": df.columns.tolist(),
        "Data Types": df.dtypes.to_dict(),
        "Missing Values": df.isnull().sum().to_dict(),
        "Basic Stats": df.describe(include='all').to_dict(), # Include non-numeric
    }

    # Format for better readability, especially for LLM
    profile_str = "Dataset Profile:\n"
    profile_str += f"- Shape: {profile['Shape']}\n"
    profile_str += f"- Columns: {profile['Columns']}\n"
    profile_str += "- Data Types:\n"
    for col, dtype in profile['Data Types'].items():
        profile_str += f"  - {col}: {dtype}\n"
    profile_str += "- Missing Values (Count):\n"
    for col, count in profile['Missing Values'].items():
        if count > 0:
            profile_str += f"  - {col}: {count}\n"
    if not any(profile['Missing Values'].values()):
         profile_str += "  - No missing values detected.\n"

    # Add basic stats summary
    profile_str += "\n- Basic Statistics:\n"
    # Convert describe() output to a more readable string format
    try:
        stats_df = df.describe(include='all')
        profile_str += stats_df.to_string()
    except Exception as e:
        profile_str += f"  - Could not generate descriptive statistics ({e}).\n"

    # Add correlation info for numeric columns only
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        profile_str += "\n\n- Correlation Matrix (Numeric Columns):\n"
        try:
            corr_matrix = numeric_df.corr()
            profile_str += corr_matrix.to_string()
        except Exception as e:
            profile_str += f"  - Could not compute correlation matrix ({e}).\n"

    return profile_str
