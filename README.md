# 📊 Autonomous Data Visualization Agent

This Streamlit application leverages AI agents (built with Langchain/Langgraph) to automatically analyze uploaded CSV data and generate relevant visualizations.

## 🎥 Preview

<video width="100%" controls>
  <source src="assets/preview-video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## ✨ Features

*   **CSV Upload:** Easily upload your data via a simple interface.
*   **AI-Powered Analysis:** An autonomous agent analyzes the data, profiles it, and brainstorms visualization ideas.
*   **Automatic Visualization Generation:** Generates charts based on the AI's analysis.
*   **Code & Explanation:** Displays the Python code used to generate each chart along with an explanation.
*   **Interactive UI:** Built with Streamlit for easy interaction.
*   **Customizable:** Control the maximum number of visualizations to generate.

## 🛠️ Tech Stack

*   **Frontend:** Streamlit
*   **AI/Agent Framework:** Langchain, Langgraph
*   **LLMs:** Google Gemini / OpenAI (requires API keys)
*   **Data Handling:** Pandas
*   **Plotting:** Plotly, Seaborn
*   **Environment Management:** dotenv
*   **Language:** Python 3

## 🚀 Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/arnnv/ai-analytics.git
    cd ai-analytics
    ```

2.  **Create a virtual environment:** (Recommended)
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    *   Using pip:
        ```bash
        pip install -r requirements.txt
        ```
    *   *Alternatively, if you use `uv`*:
        ```bash
        uv pip sync uv.lock
        ```

4.  **Set up Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your API keys for the required AI models (e.g., `GOOGLE_API_KEY`, `OPENAI_API_KEY`).

## ▶️ Running the Application

Once the setup is complete, run the Streamlit application:

```bash
streamlit run main.py
```

This will start the application, and you can access it in your web browser at the provided local URL.

## 📚 Usage

1.  Open the application in your browser.
2.  Use the sidebar to upload your CSV file.
3.  Adjust the slider to set the maximum number of visualizations you want.
4.  Click the "Analyze Data" button.
5.  The agent will process the data and display the generated visualizations along with code and explanations in the main area.

## 📁 Project Structure

*   `main.py`: The main Streamlit application script.
*   `graph.py`: Contains the Langgraph agent definition and logic.
*   `utils.py`: Utility functions (e.g., data loading).
*   `requirements.txt` / `uv.lock`: Project dependencies.
*   `.env.example`: Template for environment variables (API keys).
*   `README.md`: This file.