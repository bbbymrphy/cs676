"""
Gradio Frontend for Time Series Analysis with Claude AI Chatbot
Features:
- Interactive chatbot powered by Claude AI
- File upload for time series data
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
import base64
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import the time series analyzer
from time_series_analyzer import TimeSeriesAnalyzer

# Initialize Claude client
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("Warning: ANTHROPIC_API_KEY not found in environment variables")
    print("Please set it with: export ANTHROPIC_API_KEY='your-api-key'")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Global state to maintain analyzer instance and conversation history
class AnalysisState:
    def __init__(self):
        self.analyzer = None
        self.data_loaded = False
        self.models_fitted = False
        self.conversation_history = []
        self.last_analysis_steps = []
        self.available_plots = []

    def reset(self):
        self.analyzer = None
        self.data_loaded = False
        self.models_fitted = False
        self.last_analysis_steps = []
        self.available_plots = []

state = AnalysisState()

# Tool definitions for Claude
TOOLS = [
    {
        "name": "load_data",
        "description": "Load time series data from a CSV file. Use this when the user wants to load or analyze their data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the CSV file"
                },
                "date_column": {
                    "type": "string",
                    "description": "Name of the date column (optional, will auto-detect if not provided)"
                },
                "value_column": {
                    "type": "string",
                    "description": "Name of the value column (optional, will auto-detect if not provided)"
                },
                "freq": {
                    "type": "string",
                    "description": "Frequency of the time series: 'D' for daily, 'M' for monthly, 'W' for weekly, etc. Default is 'D'"
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "generate_sample_data",
        "description": "Generate sample time series data for testing or demonstration. Use this when the user wants to see a demo or doesn't have data yet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n_periods": {
                    "type": "integer",
                    "description": "Number of time periods to generate (default: 365)"
                },
                "trend": {
                    "type": "string",
                    "description": "Type of trend: 'linear', 'exponential', or 'none' (default: 'linear')"
                },
                "seasonality": {
                    "type": "boolean",
                    "description": "Whether to include seasonality (default: true)"
                },
                "freq": {
                    "type": "string",
                    "description": "Frequency: 'D' for daily, 'M' for monthly (default: 'D')"
                }
            },
            "required": []
        }
    },
    {
        "name": "explore_data",
        "description": "Perform exploratory data analysis including plots and summary statistics. Use this when user wants to understand or visualize their data.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "test_stationarity",
        "description": "Test if the time series is stationary using ADF and KPSS tests. Use this when user asks about stationarity or wants statistical tests.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "plot_acf_pacf",
        "description": "Plot autocorrelation and partial autocorrelation functions. Use this when user wants to see ACF/PACF or determine ARIMA parameters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "lags": {
                    "type": "integer",
                    "description": "Number of lags to plot (default: 40)"
                }
            },
            "required": []
        }
    },
    {
        "name": "decompose",
        "description": "Decompose time series into trend, seasonal, and residual components. Use when user wants to see decomposition or understand components.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Decomposition model: 'additive' or 'multiplicative' (default: 'additive')"
                },
                "period": {
                    "type": "integer",
                    "description": "Seasonal period (auto-detected if not provided)"
                }
            },
            "required": []
        }
    },
    {
        "name": "split_data",
        "description": "Split data into training and testing sets. Must be done before fitting models.",
        "input_schema": {
            "type": "object",
            "properties": {
                "test_size": {
                    "type": "number",
                    "description": "Proportion of data for testing (default: 0.2)"
                }
            },
            "required": []
        }
    },
    {
        "name": "fit_arima",
        "description": "Fit an ARIMA model to the time series data. Use when user wants to build a forecasting model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "p": {
                    "type": "integer",
                    "description": "AR order (default: 1)"
                },
                "d": {
                    "type": "integer",
                    "description": "Differencing order (default: 1)"
                },
                "q": {
                    "type": "integer",
                    "description": "MA order (default: 1)"
                },
                "name": {
                    "type": "string",
                    "description": "Model name (default: 'ARIMA')"
                }
            },
            "required": []
        }
    },
    {
        "name": "fit_sarima",
        "description": "Fit a SARIMA (Seasonal ARIMA) model. Use when data has seasonality.",
        "input_schema": {
            "type": "object",
            "properties": {
                "p": {"type": "integer", "description": "AR order (default: 1)"},
                "d": {"type": "integer", "description": "Differencing order (default: 1)"},
                "q": {"type": "integer", "description": "MA order (default: 1)"},
                "P": {"type": "integer", "description": "Seasonal AR order (default: 1)"},
                "D": {"type": "integer", "description": "Seasonal differencing (default: 1)"},
                "Q": {"type": "integer", "description": "Seasonal MA order (default: 1)"},
                "s": {"type": "integer", "description": "Seasonal period (default: 12)"},
                "name": {"type": "string", "description": "Model name (default: 'SARIMA')"}
            },
            "required": []
        }
    },
    {
        "name": "fit_exponential_smoothing",
        "description": "Fit an Exponential Smoothing model. Good for data with trends and seasonality.",
        "input_schema": {
            "type": "object",
            "properties": {
                "seasonal": {
                    "type": "string",
                    "description": "Seasonal component: 'add' or 'mul' (default: 'add')"
                },
                "seasonal_periods": {
                    "type": "integer",
                    "description": "Number of periods in season (default: 12)"
                },
                "name": {
                    "type": "string",
                    "description": "Model name (default: 'ExpSmoothing')"
                }
            },
            "required": []
        }
    },
    {
        "name": "fit_prophet",
        "description": "Fit a Prophet model (Facebook's forecasting tool). Great for data with multiple seasonalities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Model name (default: 'Prophet')"
                }
            },
            "required": []
        }
    },
    {
        "name": "compare_models",
        "description": "Compare all fitted models and show their performance metrics. Use after fitting multiple models.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "plot_forecasts",
        "description": "Plot forecasts from all fitted models. Use to visualize predictions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "show_train": {
                    "type": "boolean",
                    "description": "Whether to show training data (default: true)"
                }
            },
            "required": []
        }
    },
    {
        "name": "plot_residuals",
        "description": "Plot residual diagnostics for a model. Use to check model assumptions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Name of the model to analyze (uses best model if not specified)"
                }
            },
            "required": []
        }
    },
    {
        "name": "generate_report",
        "description": "Generate a comprehensive analysis report. Use when user wants a summary or final report.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Output filename (default: 'time_series_report.txt')"
                }
            },
            "required": []
        }
    },
    {
        "name": "forecast_future",
        "description": "Forecast future values beyond the dataset. Use when user wants predictions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "Number of steps to forecast (default: 30)"
                },
                "model_name": {
                    "type": "string",
                    "description": "Model to use for forecasting (uses best model if not specified)"
                }
            },
            "required": []
        }
    },
    {
        "name": "run_complete_analysis",
        "description": "Run a complete end-to-end analysis pipeline including all steps. Use when user wants a full analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "test_size": {
                    "type": "number",
                    "description": "Test set proportion (default: 0.2)"
                },
                "forecast_steps": {
                    "type": "integer",
                    "description": "Steps to forecast into future (default: 30)"
                }
            },
            "required": []
        }
    }
]

def execute_tool(tool_name, tool_input):
    """Execute a tool and return the result"""
    try:
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        result = {"success": True, "output": "", "plots": []}

        if tool_name == "load_data":
            if state.analyzer is None:
                state.analyzer = TimeSeriesAnalyzer(freq=tool_input.get('freq', 'D'))
            state.analyzer.load_data(
                tool_input['file_path'],
                tool_input.get('date_column'),
                tool_input.get('value_column')
            )
            state.data_loaded = True
            result["output"] = "Data loaded successfully"

        elif tool_name == "generate_sample_data":
            if state.analyzer is None:
                state.analyzer = TimeSeriesAnalyzer(freq=tool_input.get('freq', 'D'))
            state.analyzer.generate_sample_data(
                n_periods=tool_input.get('n_periods', 365),
                trend=tool_input.get('trend', 'linear'),
                seasonality=tool_input.get('seasonality', True)
            )
            state.data_loaded = True
            result["output"] = "Sample data generated successfully"

        elif tool_name == "explore_data":
            if not state.data_loaded:
                raise Exception("No data loaded. Please load data first.")
            state.analyzer.explore_data()
            result["plots"].append("01_exploratory_analysis.png")

        elif tool_name == "test_stationarity":
            if not state.data_loaded:
                raise Exception("No data loaded. Please load data first.")
            state.analyzer.test_stationarity()

        elif tool_name == "plot_acf_pacf":
            if not state.data_loaded:
                raise Exception("No data loaded. Please load data first.")
            state.analyzer.plot_acf_pacf(lags=tool_input.get('lags', 40))
            result["plots"].append("02_acf_pacf.png")

        elif tool_name == "decompose":
            if not state.data_loaded:
                raise Exception("No data loaded. Please load data first.")
            state.analyzer.decompose(
                model=tool_input.get('model', 'additive'),
                period=tool_input.get('period')
            )
            result["plots"].append("03_decomposition.png")

        elif tool_name == "split_data":
            if not state.data_loaded:
                raise Exception("No data loaded. Please load data first.")
            state.analyzer.split_data(test_size=tool_input.get('test_size', 0.2))

        elif tool_name == "fit_arima":
            if not state.data_loaded:
                raise Exception("No data loaded. Please load data first.")
            if state.analyzer.train is None:
                state.analyzer.split_data(test_size=0.2)

            order = (
                tool_input.get('p', 1),
                tool_input.get('d', 1),
                tool_input.get('q', 1)
            )
            state.analyzer.fit_arima(order=order, name=tool_input.get('name', 'ARIMA'))
            state.models_fitted = True

        elif tool_name == "fit_sarima":
            if not state.data_loaded:
                raise Exception("No data loaded. Please load data first.")
            if state.analyzer.train is None:
                state.analyzer.split_data(test_size=0.2)

            order = (
                tool_input.get('p', 1),
                tool_input.get('d', 1),
                tool_input.get('q', 1)
            )
            seasonal_order = (
                tool_input.get('P', 1),
                tool_input.get('D', 1),
                tool_input.get('Q', 1),
                tool_input.get('s', 12)
            )
            state.analyzer.fit_sarima(
                order=order,
                seasonal_order=seasonal_order,
                name=tool_input.get('name', 'SARIMA')
            )
            state.models_fitted = True

        elif tool_name == "fit_exponential_smoothing":
            if not state.data_loaded:
                raise Exception("No data loaded. Please load data first.")
            if state.analyzer.train is None:
                state.analyzer.split_data(test_size=0.2)

            state.analyzer.fit_exponential_smoothing(
                seasonal=tool_input.get('seasonal', 'add'),
                seasonal_periods=tool_input.get('seasonal_periods', 12),
                name=tool_input.get('name', 'ExpSmoothing')
            )
            state.models_fitted = True

        elif tool_name == "fit_prophet":
            if not state.data_loaded:
                raise Exception("No data loaded. Please load data first.")
            if state.analyzer.train is None:
                state.analyzer.split_data(test_size=0.2)

            state.analyzer.fit_prophet(name=tool_input.get('name', 'Prophet'))
            state.models_fitted = True

        elif tool_name == "compare_models":
            if not state.models_fitted:
                raise Exception("No models fitted yet. Please fit models first.")
            state.analyzer.compare_models()

        elif tool_name == "plot_forecasts":
            if not state.models_fitted:
                raise Exception("No models fitted yet. Please fit models first.")
            state.analyzer.plot_forecasts(show_train=tool_input.get('show_train', True))
            result["plots"].append("04_forecasts.png")

        elif tool_name == "plot_residuals":
            if not state.models_fitted:
                raise Exception("No models fitted yet. Please fit models first.")
            state.analyzer.plot_residuals(model_name=tool_input.get('model_name'))
            result["plots"].append("05_residuals.png")

        elif tool_name == "generate_report":
            if not state.data_loaded:
                raise Exception("No data loaded. Please load data first.")
            state.analyzer.generate_report(filename=tool_input.get('filename', 'time_series_report.txt'))

        elif tool_name == "forecast_future":
            if not state.models_fitted:
                raise Exception("No models fitted yet. Please fit models first.")
            state.analyzer.forecast_future(
                steps=tool_input.get('steps', 30),
                model_name=tool_input.get('model_name')
            )
            result["plots"].append("06_future_forecast.png")

        elif tool_name == "run_complete_analysis":
            if not state.data_loaded:
                raise Exception("No data loaded. Please load data or generate sample data first.")

            # Run complete pipeline
            state.analyzer.explore_data()
            state.analyzer.test_stationarity()
            state.analyzer.plot_acf_pacf(lags=40)
            state.analyzer.decompose(model='additive')
            state.analyzer.split_data(test_size=tool_input.get('test_size', 0.2))
            state.analyzer.fit_arima(order=(2, 1, 2), name='ARIMA(2,1,2)')
            state.analyzer.fit_sarima(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), name='SARIMA')
            state.analyzer.fit_exponential_smoothing(seasonal='add', seasonal_periods=7, name='ExpSmoothing')

            try:
                state.analyzer.fit_prophet(name='Prophet')
            except:
                pass

            state.models_fitted = True
            state.analyzer.compare_models()
            state.analyzer.plot_forecasts(show_train=True)
            state.analyzer.plot_residuals()
            state.analyzer.generate_report('time_series_report.txt')
            state.analyzer.forecast_future(steps=tool_input.get('forecast_steps', 30))

            result["plots"] = [
                "01_exploratory_analysis.png",
                "02_acf_pacf.png",
                "03_decomposition.png",
                "04_forecasts.png",
                "05_residuals.png",
                "06_future_forecast.png"
            ]
            result["output"] = "Complete analysis finished! All plots and reports generated."

        # Restore stdout and get captured output
        sys.stdout = old_stdout
        console_output = captured_output.getvalue()

        if console_output:
            result["output"] = console_output

        # Update available plots
        for plot in result["plots"]:
            if plot not in state.available_plots:
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
    """Handle chat with Claude and execute time series analysis"""

    if not client:
        yield "Error: ANTHROPIC_API_KEY not set. Please configure your API key."
        return

    # Handle file upload
    if uploaded_file is not None:
        file_path = uploaded_file.name
        # Initialize analyzer with uploaded file
        try:
            df = pd.read_csv(file_path)
            # Auto-detect columns
            date_col = None
            value_col = None

            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_col = col
                elif df[col].dtype in ['float64', 'int64'] and value_col is None:
                    value_col = col

            if state.analyzer is None:
                state.analyzer = TimeSeriesAnalyzer(freq='D')

            state.analyzer.load_data(file_path, date_col, value_col)
            state.data_loaded = True

            message = f"I've uploaded a file: {Path(file_path).name}. {message}"
        except Exception as e:
            yield f"Error loading file: {str(e)}"
            return

    # Build conversation for Claude
    messages = []

    # Add conversation history
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})

    # Add current message
    messages.append({"role": "user", "content": message})

    # System prompt
    system_prompt = """You are a helpful AI assistant specializing in time series analysis. You have access to various tools to perform comprehensive time series analysis including data loading, exploration, statistical testing, decomposition, modeling (ARIMA, SARIMA, Prophet, Exponential Smoothing), and forecasting.

When users ask you to analyze data:
1. First check if data is loaded (you can generate sample data or ask them to upload a file)
2. Then perform the appropriate analysis steps
3. Be proactive - suggest useful analyses based on what they ask for
4. Explain the results in a clear, understandable way
5. If they ask for a "complete analysis" or "full analysis", use the run_complete_analysis tool

Current state:
- Data loaded: """ + str(state.data_loaded) + """
- Models fitted: """ + str(state.models_fitted) + """

Be conversational and helpful. Explain what you're doing and why."""

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

        # Process response with tool use loop
        while response.stop_reason == "tool_use":
            # Extract tool uses and text
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
                    if result["output"]:
                        response_text += f"```\n{result['output'][:500]}...\n```\n"
                else:
                    response_text += f"✗ Error in {tool_use.name}: {result['output']}\n"

                yield response_text

            # Continue conversation with tool results
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
            response_text += "\n".join(final_text)

        yield response_text

    except Exception as e:
        yield f"Error communicating with Claude: {str(e)}"

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
    return "Analysis state reset. Ready for new analysis!"

# Create Gradio interface
with gr.Blocks(title="Time Series Analysis with Claude AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📊 Time Series Analysis with Claude AI

    Upload your time series data or generate sample data, then chat with Claude to perform comprehensive analysis!
    Claude can help you with:
    - Exploratory data analysis
    - Stationarity testing
    - Time series decomposition
    - Multiple forecasting models (ARIMA, SARIMA, Prophet, Exponential Smoothing)
    - Model comparison and evaluation
    - Future forecasting

    **Try asking:**
    - "Generate sample data and run a complete analysis"
    - "Test if my data is stationary"
    - "Fit an ARIMA model and show me the forecast"
    - "Compare all available models"
    """)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, label="Chat with Claude")
            msg = gr.Textbox(
                label="Your message",
                placeholder="Ask Claude to analyze your time series data...",
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
            gr.Markdown("### 📈 Generated Plots")
            plot_selector = gr.Dropdown(
                choices=[],
                label="Select Plot to View",
                interactive=True
            )
            plot_display = gr.Image(label="Plot Viewer", type="pil")

            refresh_plots = gr.Button("🔄 Refresh Plots List")

            gr.Markdown("### 📁 Download Results")
            report_download = gr.File(label="Analysis Report", visible=False)
            forecast_download = gr.File(label="Forecast CSV", visible=False)

    # Event handlers
    def respond(message, chat_history, file):
        if not message.strip() and file is None:
            return "", chat_history

        bot_response = ""
        for partial_response in chat_with_claude(message, chat_history, file):
            bot_response = partial_response

        chat_history.append((message, bot_response))
        return "", chat_history

    def update_plot_list():
        plots = get_available_plots()
        return gr.Dropdown(choices=plots, value=plots[0] if plots else None)

    def check_files():
        report_visible = os.path.exists("time_series_report.txt")
        forecast_visible = os.path.exists("future_forecast.csv")

        return (
            gr.File(value="time_series_report.txt" if report_visible else None, visible=report_visible),
            gr.File(value="future_forecast.csv" if forecast_visible else None, visible=forecast_visible)
        )

    submit.click(respond, [msg, chatbot, file_upload], [msg, chatbot]).then(
        update_plot_list, None, plot_selector
    ).then(
        check_files, None, [report_download, forecast_download]
    )

    msg.submit(respond, [msg, chatbot, file_upload], [msg, chatbot]).then(
        update_plot_list, None, plot_selector
    ).then(
        check_files, None, [report_download, forecast_download]
    )

    clear.click(lambda: None, None, chatbot, queue=False)
    reset.click(reset_analysis, None, msg)
    refresh_plots.click(update_plot_list, None, plot_selector)
    plot_selector.change(display_plot, plot_selector, plot_display)

    # Example prompts
    gr.Examples(
        examples=[
            "Generate sample data with 500 days of data and show me exploratory analysis",
            "Test if the data is stationary and explain the results",
            "Run a complete time series analysis on this data",
            "Fit an ARIMA model with p=2, d=1, q=2",
            "Compare all available forecasting models",
            "Forecast the next 60 days using the best model",
            "Show me the ACF and PACF plots",
            "Decompose the time series into trend, seasonal, and residual components"
        ],
        inputs=msg
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
