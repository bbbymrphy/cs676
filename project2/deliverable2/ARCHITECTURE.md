# System Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│                      (Gradio Web App)                           │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Chatbot   │  │ File Upload  │  │   Plot Viewer        │   │
│  │  Interface │  │              │  │                      │   │
│  └────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Claude AI (Anthropic)                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  - Natural Language Understanding                        │  │
│  │  - Tool Selection & Orchestration                        │  │
│  │  - Result Interpretation & Explanation                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Tool Execution Layer                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  17 Available Tools:                                   │    │
│  │  • load_data              • fit_arima                  │    │
│  │  • generate_sample_data   • fit_sarima                 │    │
│  │  • explore_data           • fit_exponential_smoothing  │    │
│  │  • test_stationarity      • fit_prophet                │    │
│  │  • plot_acf_pacf          • compare_models             │    │
│  │  • decompose              • plot_forecasts             │    │
│  │  • split_data             • plot_residuals             │    │
│  │  • generate_report        • forecast_future            │    │
│  │  • run_complete_analysis                               │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 TimeSeriesAnalyzer (Core Engine)                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Data Processing:                                        │  │
│  │  • pandas - Data manipulation                           │  │
│  │  • numpy - Numerical operations                         │  │
│  │                                                          │  │
│  │  Statistical Testing:                                    │  │
│  │  • statsmodels - ADF, KPSS, Ljung-Box tests            │  │
│  │  • scipy - Statistical distributions                    │  │
│  │                                                          │  │
│  │  Decomposition & Analysis:                              │  │
│  │  • statsmodels - Seasonal decomposition                 │  │
│  │  • statsmodels - ACF/PACF plots                         │  │
│  │                                                          │  │
│  │  Forecasting Models:                                     │  │
│  │  • ARIMA (statsmodels)                                  │  │
│  │  • SARIMA (statsmodels)                                 │  │
│  │  • Exponential Smoothing (statsmodels)                  │  │
│  │  • Prophet (Facebook)                                    │  │
│  │                                                          │  │
│  │  Visualization:                                          │  │
│  │  • matplotlib - Plotting                                │  │
│  │  • seaborn - Statistical visualizations                 │  │
│  │                                                          │  │
│  │  Evaluation:                                             │  │
│  │  • scikit-learn - RMSE, MAE, R²                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          Outputs                                │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐  │
│  │  PNG Images    │  │  Text Reports  │  │  CSV Forecasts  │  │
│  │  (6 plots)     │  │  (.txt)        │  │  (.csv)         │  │
│  └────────────────┘  └────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. User Interaction
```
User Message → Gradio Interface → Claude AI
```

### 2. Claude Processing
```
Message Analysis → Tool Selection → Tool Parameters
```

### 3. Tool Execution
```
Tool Call → TimeSeriesAnalyzer Method → Statistical Computation
```

### 4. Result Generation
```
Computation → Plots/Reports → File System
```

### 5. Response
```
Files → Gradio Display → User sees results
Claude Explanation → Chatbot → User understands
```

## Component Details

### Gradio Frontend (`gradio_app.py`)

**Responsibilities:**
- Serve web interface
- Manage file uploads
- Route user messages to Claude
- Execute tools based on Claude's decisions
- Display plots and results
- Maintain conversation state

**Key Features:**
- Real-time chat interface
- File upload support
- Plot selector and viewer
- Download links for reports
- Example prompts
- State management

### Claude AI Integration

**Capabilities:**
- Understand natural language queries
- Select appropriate analysis tools
- Determine optimal parameters
- Chain multiple operations
- Explain results to users
- Provide insights and recommendations

**Tool Use Loop:**
```
1. Receive user message
2. Analyze intent
3. Select tool(s) to use
4. Execute tool(s)
5. Receive results
6. Formulate response
7. Return to user
```

### TimeSeriesAnalyzer (`time_series_analyzer.py`)

**Class Structure:**
```python
TimeSeriesAnalyzer
├── __init__()
├── Data Loading
│   ├── load_data()
│   └── generate_sample_data()
├── Exploration
│   ├── explore_data()
│   └── test_stationarity()
├── Visualization
│   ├── plot_acf_pacf()
│   └── decompose()
├── Preparation
│   └── split_data()
├── Modeling
│   ├── fit_arima()
│   ├── fit_sarima()
│   ├── fit_exponential_smoothing()
│   └── fit_prophet()
├── Evaluation
│   ├── _calculate_metrics()
│   ├── compare_models()
│   ├── plot_forecasts()
│   └── plot_residuals()
└── Output
    ├── generate_report()
    └── forecast_future()
```

**Internal State:**
```python
{
    'data': pd.Series,           # Time series data
    'train': pd.Series,          # Training set
    'test': pd.Series,           # Test set
    'models': dict,              # Fitted models
    'forecasts': dict,           # Model predictions
    'metrics': dict,             # Performance metrics
    'decomposition': object      # Decomposition results
}
```

## Tool Definitions

Each tool has:
- **Name**: Identifier for Claude
- **Description**: When Claude should use it
- **Input Schema**: Required/optional parameters
- **Implementation**: Python function that executes

Example:
```python
{
    "name": "fit_arima",
    "description": "Fit an ARIMA model...",
    "input_schema": {
        "type": "object",
        "properties": {
            "p": {"type": "integer"},
            "d": {"type": "integer"},
            "q": {"type": "integer"}
        }
    }
}
```

## Error Handling

### Three Levels:
1. **Tool Level**: Try-catch in execute_tool()
2. **Analyzer Level**: Validation in TimeSeriesAnalyzer methods
3. **Claude Level**: Claude interprets errors and explains to user

### Flow:
```
Error Occurs → Caught by Tool → Returned to Claude →
Claude Explains → User Understands → Suggests Fix
```

## Performance Considerations

### Optimization Strategies:
1. **Non-blocking UI**: Gradio uses async/streaming
2. **Matplotlib Backend**: Use 'Agg' (non-interactive)
3. **Model Caching**: Store fitted models in state
4. **Plot Reuse**: Save plots to disk, display from files
5. **Streaming Responses**: Show Claude's thinking in real-time

### Scalability:
- Single user instance (state management)
- File-based plot storage
- Memory-efficient pandas operations
- Model fitting can be slow for large datasets

## Security Considerations

### API Key Management:
- Environment variable (not hardcoded)
- Never logged or exposed
- User responsible for their key

### File Handling:
- CSV upload sanitization
- Local file system only
- No remote file access
- Temporary files cleaned up

### Tool Execution:
- Sandboxed to TimeSeriesAnalyzer methods
- No arbitrary code execution
- No system commands
- No file system manipulation outside working directory

## Extension Points

### Adding New Tools:
1. Define tool in TOOLS array
2. Add case in execute_tool()
3. Implement in TimeSeriesAnalyzer (if needed)
4. Claude automatically learns to use it!

### Adding New Models:
1. Create fit_<model_name>() method
2. Follow existing pattern (store model, forecast, metrics)
3. Add tool definition
4. No other changes needed!

### Customizing UI:
- Modify Gradio blocks in gradio_app.py
- Add new components
- Change theme
- Add tabs/sections

## Dependencies Graph

```
gradio_app.py
├── gradio
├── anthropic
├── pandas
├── PIL (Pillow)
└── time_series_analyzer.py
    ├── pandas
    ├── numpy
    ├── matplotlib
    ├── seaborn
    ├── statsmodels
    │   ├── tsa.stattools (ADF, KPSS)
    │   ├── stats.diagnostic (Ljung-Box)
    │   ├── tsa.seasonal (decompose)
    │   ├── graphics.tsaplots (ACF/PACF)
    │   ├── tsa.arima (ARIMA)
    │   ├── tsa.statespace (SARIMA)
    │   └── tsa.holtwinters (Exp Smoothing)
    ├── sklearn.metrics
    ├── scipy.stats
    └── prophet (optional)
```

## Deployment Options

### Local Development:
```bash
python gradio_app.py
# Access at localhost:7860
```

### Gradio Share:
```python
demo.launch(share=True)  # Creates public URL
```

### Docker:
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "gradio_app.py"]
```

### Cloud Platforms:
- Hugging Face Spaces
- Google Cloud Run
- AWS Lambda (with API Gateway)
- Azure Container Instances

## Future Enhancements

Potential additions:
- Multi-user support with session management
- Database storage for analyses
- More forecasting models (LSTM, XGBoost)
- Real-time data streaming
- Automated model selection (auto-ARIMA)
- Interactive plot editing
- Export to PowerPoint/PDF
- Scheduling and alerts
- API endpoint exposure
