"""
Unified Gradio Frontend for Machine Learning Analysis with Claude AI Chatbot
Features:
- Intelligent tool selection (Time Series vs Random Forest)
- Interactive chatbot powered by Claude AI
- File upload for data
- Automated analysis execution based on natural language commands
- Image display of generated plots
- Download results and reports
"""

import gradio as gr
import anthropic
import os
import json
import io
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib
import seaborn as sns
matplotlib.use('Agg')  # Use non-interactive backend

# Import analyzers
from time_series_analyzer import TimeSeriesAnalyzer
from random_forest_analyzer import RandomForestAnalyzer

# Initialize Claude client
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("Warning: ANTHROPIC_API_KEY not found in environment variables")
    print("Please set it with: export ANTHROPIC_API_KEY='your-api-key'")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None


# Global state to maintain analyzer instances
class UnifiedAnalysisState:
    def __init__(self):
        self.ts_analyzer = None  # Time series analyzer
        self.rf_analyzer = None  # Random forest analyzer
        self.current_mode = None  # 'timeseries' or 'randomforest'
        self.data_loaded = False
        self.models_fitted = False
        self.available_plots = []

    def reset(self):
        self.ts_analyzer = None
        self.rf_analyzer = None
        self.current_mode = None
        self.data_loaded = False
        self.models_fitted = False
        self.available_plots = []


state = UnifiedAnalysisState()


# Combined tool definitions for Claude
TOOLS = [
    # Meta tool for analysis type selection
    {
        "name": "select_analysis_type",
        "description": "Determine which type of analysis to perform based on the user's request. Use this FIRST to decide between time series analysis or random forest/machine learning analysis. Time series is for temporal data with dates/times. Random Forest is for classification/regression with features and a target variable.",
        "input_schema": {
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "enum": ["timeseries", "randomforest"],
                    "description": "Type of analysis: 'timeseries' for temporal forecasting, 'randomforest' for classification/regression"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why this analysis type was chosen"
                }
            },
            "required": ["analysis_type"]
        }
    },

    # Time Series Tools
    {
        "name": "ts_load_data",
        "description": "Load time series data from CSV. Use when user wants temporal analysis, forecasting, or has data with dates/timestamps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "date_column": {"type": "string"},
                "value_column": {"type": "string"},
                "freq": {"type": "string", "description": "'D' for daily, 'M' for monthly, etc."}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "ts_generate_sample",
        "description": "Generate sample time series data for demonstration.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n_periods": {"type": "integer"},
                "trend": {"type": "string"},
                "seasonality": {"type": "boolean"}
            }
        }
    },
    {
        "name": "ts_explore",
        "description": "Perform exploratory analysis on time series data.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "ts_test_stationarity",
        "description": "Test if time series is stationary using ADF and KPSS tests.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "ts_decompose",
        "description": "Decompose time series into trend, seasonal, and residual components.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "period": {"type": "integer"}
            }
        }
    },
    {
        "name": "ts_fit_arima",
        "description": "Fit ARIMA forecasting model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "p": {"type": "integer"},
                "d": {"type": "integer"},
                "q": {"type": "integer"}
            }
        }
    },
    {
        "name": "ts_forecast",
        "description": "Generate future forecasts from trained time series model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "steps": {"type": "integer"}
            }
        }
    },
    {
        "name": "ts_complete_analysis",
        "description": "Run complete end-to-end time series analysis pipeline.",
        "input_schema": {"type": "object", "properties": {}}
    },

    # Random Forest Tools
    {
        "name": "rf_load_data",
        "description": "Load data for Random Forest classification or regression. Use when user wants to predict a target variable from features, classify data, or perform supervised learning.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "target_column": {"type": "string", "description": "Column to predict"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "rf_generate_sample",
        "description": "Generate sample classification or regression data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n_samples": {"type": "integer"},
                "n_features": {"type": "integer"},
                "task": {"type": "string", "enum": ["classification", "regression"]}
            }
        }
    },
    {
        "name": "rf_explore",
        "description": "Perform exploratory data analysis on the dataset.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "rf_preprocess",
        "description": "Preprocess data: handle missing values, scale features, encode categorical variables.",
        "input_schema": {
            "type": "object",
            "properties": {
                "scale": {"type": "boolean"},
                "handle_missing": {"type": "string"}
            }
        }
    },
    {
        "name": "rf_train",
        "description": "Train Random Forest model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n_estimators": {"type": "integer"},
                "max_depth": {"type": "integer"}
            }
        }
    },
    {
        "name": "rf_evaluate",
        "description": "Evaluate Random Forest model performance.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "rf_feature_importance",
        "description": "Plot and analyze feature importance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {"type": "integer"}
            }
        }
    },
    {
        "name": "rf_cross_validate",
        "description": "Perform cross-validation on the model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "cv": {"type": "integer"}
            }
        }
    },
    {
        "name": "rf_complete_analysis",
        "description": "Run complete end-to-end Random Forest analysis pipeline.",
        "input_schema": {"type": "object", "properties": {}}
    },

    # Report generation
    {
        "name": "generate_report",
        "description": "Generate comprehensive analysis report for current analysis type.",
        "input_schema": {"type": "object", "properties": {}}
    }
]


def execute_tool(tool_name, tool_input):
    """Execute a tool and return the result"""
    try:
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        result = {"success": True, "output": "", "plots": []}

        # Meta tool: Select analysis type
        if tool_name == "select_analysis_type":
            state.current_mode = tool_input['analysis_type']
            result["output"] = f"Selected analysis mode: {state.current_mode}"
            if 'reasoning' in tool_input:
                result["output"] += f"\nReasoning: {tool_input['reasoning']}"

        # TIME SERIES TOOLS
        elif tool_name == "ts_load_data":
            if state.ts_analyzer is None:
                state.ts_analyzer = TimeSeriesAnalyzer(freq=tool_input.get('freq', 'D'))
            state.current_mode = 'timeseries'
            state.ts_analyzer.load_data(
                tool_input['file_path'],
                tool_input.get('date_column'),
                tool_input.get('value_column')
            )
            state.data_loaded = True

        elif tool_name == "ts_generate_sample":
            if state.ts_analyzer is None:
                state.ts_analyzer = TimeSeriesAnalyzer(freq='D')
            state.current_mode = 'timeseries'
            state.ts_analyzer.generate_sample_data(
                n_periods=tool_input.get('n_periods', 365),
                trend=tool_input.get('trend', 'linear'),
                seasonality=tool_input.get('seasonality', True)
            )
            state.data_loaded = True

        elif tool_name == "ts_explore":
            state.ts_analyzer.explore_data()
            result["plots"].append("01_exploratory_analysis.png")

        elif tool_name == "ts_test_stationarity":
            state.ts_analyzer.test_stationarity()

        elif tool_name == "ts_decompose":
            state.ts_analyzer.decompose(
                model=tool_input.get('model', 'additive'),
                period=tool_input.get('period')
            )
            result["plots"].append("03_decomposition.png")

        elif tool_name == "ts_fit_arima":
            if state.ts_analyzer.train is None:
                state.ts_analyzer.split_data(test_size=0.2)
            order = (
                tool_input.get('p', 1),
                tool_input.get('d', 1),
                tool_input.get('q', 1)
            )
            state.ts_analyzer.fit_arima(order=order)
            state.models_fitted = True

        elif tool_name == "ts_forecast":
            state.ts_analyzer.forecast_future(steps=tool_input.get('steps', 30))
            result["plots"].append("06_future_forecast.png")

        elif tool_name == "ts_complete_analysis":
            if not state.data_loaded:
                raise Exception("No data loaded. Please load or generate data first.")

            state.ts_analyzer.explore_data()
            state.ts_analyzer.test_stationarity()
            state.ts_analyzer.plot_acf_pacf(lags=40)
            state.ts_analyzer.decompose(model='additive')
            state.ts_analyzer.split_data(test_size=0.2)
            state.ts_analyzer.fit_arima(order=(2, 1, 2), name='ARIMA(2,1,2)')

            try:
                state.ts_analyzer.fit_sarima(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            except:
                pass

            try:
                state.ts_analyzer.fit_exponential_smoothing(seasonal='add', seasonal_periods=7)
            except:
                pass

            state.models_fitted = True
            state.ts_analyzer.compare_models()
            state.ts_analyzer.plot_forecasts(show_train=True)
            state.ts_analyzer.plot_residuals()
            state.ts_analyzer.generate_report('time_series_report.txt')
            state.ts_analyzer.forecast_future(steps=30)

            result["plots"] = [
                "01_exploratory_analysis.png",
                "02_acf_pacf.png",
                "03_decomposition.png",
                "04_forecasts.png",
                "05_residuals.png",
                "06_future_forecast.png"
            ]

        # RANDOM FOREST TOOLS
        elif tool_name == "rf_load_data":
            if state.rf_analyzer is None:
                state.rf_analyzer = RandomForestAnalyzer(task='auto')
            state.current_mode = 'randomforest'
            state.rf_analyzer.load_data(
                tool_input['file_path'],
                tool_input.get('target_column')
            )
            state.data_loaded = True

        elif tool_name == "rf_generate_sample":
            if state.rf_analyzer is None:
                state.rf_analyzer = RandomForestAnalyzer(task='auto')
            state.current_mode = 'randomforest'
            state.rf_analyzer.generate_sample_data(
                n_samples=tool_input.get('n_samples', 1000),
                n_features=tool_input.get('n_features', 10),
                task=tool_input.get('task', 'classification')
            )
            state.data_loaded = True

        elif tool_name == "rf_explore":
            state.rf_analyzer.explore_data()
            result["plots"].append("rf_01_exploratory_analysis.png")

        elif tool_name == "rf_preprocess":
            state.rf_analyzer.preprocess_data(
                handle_missing=tool_input.get('handle_missing', 'mean'),
                scale=tool_input.get('scale', True)
            )

        elif tool_name == "rf_train":
            if state.rf_analyzer.X_train is None:
                state.rf_analyzer.split_data(test_size=0.2)

            state.rf_analyzer.train_model(
                n_estimators=tool_input.get('n_estimators', 100),
                max_depth=tool_input.get('max_depth')
            )
            state.models_fitted = True

        elif tool_name == "rf_evaluate":
            state.rf_analyzer.evaluate_model()
            result["plots"].append("rf_02_evaluation.png")

        elif tool_name == "rf_feature_importance":
            state.rf_analyzer.plot_feature_importance(top_n=tool_input.get('top_n', 20))
            result["plots"].append("rf_03_feature_importance.png")

        elif tool_name == "rf_cross_validate":
            state.rf_analyzer.cross_validate(cv=tool_input.get('cv', 5))

        elif tool_name == "rf_complete_analysis":
            if not state.data_loaded:
                raise Exception("No data loaded. Please load or generate data first.")

            state.rf_analyzer.explore_data()
            state.rf_analyzer.preprocess_data(handle_missing='mean', scale=True)
            state.rf_analyzer.split_data(test_size=0.2)
            state.rf_analyzer.train_model(n_estimators=100, max_depth=10)
            state.rf_analyzer.evaluate_model()
            state.rf_analyzer.plot_feature_importance(top_n=15)
            state.rf_analyzer.cross_validate(cv=5)
            state.rf_analyzer.generate_report('random_forest_report.txt')

            state.models_fitted = True
            result["plots"] = [
                "rf_01_exploratory_analysis.png",
                "rf_02_evaluation.png",
                "rf_03_feature_importance.png"
            ]

        # GENERAL TOOLS
        elif tool_name == "generate_report":
            if state.current_mode == 'timeseries':
                state.ts_analyzer.generate_report('time_series_report.txt')
            elif state.current_mode == 'randomforest':
                state.rf_analyzer.generate_report('random_forest_report.txt')

        # Restore stdout
        sys.stdout = old_stdout
        console_output = captured_output.getvalue()

        if console_output:
            # Truncate very long outputs
            if len(console_output) > 2000:
                result["output"] = console_output[:2000] + "\n... (output truncated)"
            else:
                result["output"] = console_output

        # Update available plots
        for plot in result["plots"]:
            if plot not in state.available_plots and os.path.exists(plot):
                state.available_plots.append(plot)

        return result

    except Exception as e:
        sys.stdout = old_stdout
        return {
            "success": False,
            "output": f"Error: {str(e)}",
            "plots": []
        }


def chat_with_claude(message, history, uploaded_file):
    """Handle chat with Claude and execute analysis"""

    if not client:
        yield "Error: ANTHROPIC_API_KEY not set. Please configure your API key."
        return

    # Handle file upload
    file_path = None
    if uploaded_file is not None:
        file_path = uploaded_file.name
        message = f"I've uploaded a file: {Path(file_path).name}. {message}"

    # Build conversation
    messages = []
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})

    # System prompt
    system_prompt = f"""You are an expert data science assistant with access to both time series analysis and random forest machine learning tools.

**IMPORTANT: First determine the analysis type:**
- Use TIME SERIES for: forecasting, temporal data, dates/timestamps, ARIMA, seasonality, trends over time
- Use RANDOM FOREST for: classification, regression, feature importance, supervised learning, predicting categories or values from features

Current state:
- Analysis mode: {state.current_mode or 'Not selected'}
- Data loaded: {state.data_loaded}
- Models fitted: {state.models_fitted}

When a user asks for analysis:
1. FIRST use select_analysis_type to determine which approach to use
2. Then use the appropriate tools (ts_* for time series, rf_* for random forest)
3. Be proactive and suggest complete analyses
4. Explain results clearly

If the file is uploaded, consider its structure when selecting the analysis type."""

    try:
        response_text = ""

        # Initial API call
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )

        # Tool use loop
        max_iterations = 15
        iteration = 0

        while response.stop_reason == "tool_use" and iteration < max_iterations:
            iteration += 1

            tool_uses = [block for block in response.content if block.type == "tool_use"]
            text_blocks = [block.text for block in response.content if hasattr(block, "text")]

            if text_blocks:
                response_text += "\n".join(text_blocks) + "\n\n"
                yield response_text

            # Execute tools
            tool_results = []
            for tool_use in tool_uses:
                result = execute_tool(tool_use.name, tool_use.input)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(result)
                })

                # Show progress
                if result["success"]:
                    response_text += f"✓ Executed: {tool_use.name}\n"
                else:
                    response_text += f"✗ Error in {tool_use.name}: {result['output']}\n"

                yield response_text

            # Continue conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                system=system_prompt,
                tools=TOOLS,
                messages=messages
            )

        # Final response
        final_text = [block.text for block in response.content if hasattr(block, "text")]
        if final_text:
            response_text += "\n" + "\n".join(final_text)

        yield response_text

    except Exception as e:
        yield f"Error: {str(e)}"


def get_available_plots():
    """Return list of available plot files"""
    plots = []
    for plot_file in state.available_plots:
        if os.path.exists(plot_file):
            plots.append(plot_file)
    return plots


def display_plot(plot_name):
    """Display a specific plot"""
    if plot_name and os.path.exists(plot_name):
        return Image.open(plot_name)
    return None


def reset_analysis():
    """Reset the analysis state"""
    state.reset()
    return "Analysis state reset!"


# Create Gradio interface
with gr.Blocks(title="ML Analysis with Claude AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Machine Learning Analysis with Claude AI

    **Intelligent tool selection:** Claude automatically chooses between Time Series Analysis and Random Forest based on your data and request!

    ## 📊 Time Series Analysis
    Perfect for: Forecasting, temporal data, ARIMA, seasonality, trend analysis

    ## 🌲 Random Forest
    Perfect for: Classification, regression, feature importance, supervised learning

    **Try asking:**
    - "Generate sample time series data and forecast the next 30 days"
    - "Create a classification dataset and train a random forest model"
    - "I have sales data over time, help me forecast future sales"
    - "I want to predict customer churn from features"
    """)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, label="Chat with Claude")
            msg = gr.Textbox(
                label="Your message",
                placeholder="Describe what you want to analyze...",
                lines=2
            )
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear Chat")
                reset = gr.Button("Reset Analysis", variant="stop")

            file_upload = gr.File(
                label="Upload CSV File (optional)",
                file_types=[".csv"],
                type="filepath"
            )

        with gr.Column(scale=1):
            mode_display = gr.Textbox(
                label="Current Analysis Mode",
                value="Not selected",
                interactive=False
            )

            gr.Markdown("### 📈 Generated Plots")
            plot_selector = gr.Dropdown(
                choices=[],
                label="Select Plot to View",
                interactive=True
            )
            plot_display = gr.Image(label="Plot Viewer", type="pil")

            refresh_plots = gr.Button("🔄 Refresh Plots")

            gr.Markdown("### 📁 Download Results")
            report_download = gr.File(label="Analysis Report", visible=False)

    # Event handlers
    def respond(message, chat_history, file):
        if not message.strip() and file is None:
            return "", chat_history, state.current_mode or "Not selected"

        bot_response = ""
        for partial_response in chat_with_claude(message, chat_history, file):
            bot_response = partial_response

        chat_history.append((message, bot_response))
        return "", chat_history, state.current_mode or "Not selected"

    def update_plot_list():
        plots = get_available_plots()
        return gr.Dropdown(choices=plots, value=plots[0] if plots else None)

    def check_files():
        # Check for both types of reports
        ts_report = os.path.exists("time_series_report.txt")
        rf_report = os.path.exists("random_forest_report.txt")

        if ts_report:
            return gr.File(value="time_series_report.txt", visible=True)
        elif rf_report:
            return gr.File(value="random_forest_report.txt", visible=True)
        else:
            return gr.File(visible=False)

    submit.click(respond, [msg, chatbot, file_upload], [msg, chatbot, mode_display]).then(
        update_plot_list, None, plot_selector
    ).then(
        check_files, None, report_download
    )

    msg.submit(respond, [msg, chatbot, file_upload], [msg, chatbot, mode_display]).then(
        update_plot_list, None, plot_selector
    ).then(
        check_files, None, report_download
    )

    clear.click(lambda: None, None, chatbot, queue=False)
    reset.click(reset_analysis, None, msg)
    refresh_plots.click(update_plot_list, None, plot_selector)
    plot_selector.change(display_plot, plot_selector, plot_display)

    # Example prompts
    gr.Examples(
        examples=[
            # Time Series examples
            "Generate sample time series data and run a complete analysis",
            "I have daily sales data, help me forecast the next 60 days",
            "Test if my time series is stationary and decompose it",

            # Random Forest examples
            "Create a classification dataset with 1000 samples and train a random forest",
            "I have customer data, help me predict which customers will churn",
            "Generate regression data and show me feature importance",

            # General
            "What's the difference between time series and random forest analysis?",
            "Run a complete analysis on my uploaded data"
        ],
        inputs=msg
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True
    )
