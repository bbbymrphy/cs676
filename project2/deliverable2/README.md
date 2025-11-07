# Time Series Analysis with Claude AI Chatbot

An interactive Gradio web application that combines comprehensive time series analysis with Claude AI's natural language understanding. Chat with Claude to perform complex time series analyses without writing code!

## Features

### 🤖 Claude AI Chatbot Interface
- Natural language interaction for time series analysis
- Intelligent tool selection based on user requests
- Conversational explanations of results
- Proactive suggestions for analysis steps

### 📊 Comprehensive Time Series Analysis
- **Data Loading**: CSV upload or sample data generation
- **Exploratory Analysis**: Summary statistics, distributions, visualizations
- **Statistical Testing**:
  - Augmented Dickey-Fuller (ADF) test
  - KPSS test
  - Ljung-Box test for autocorrelation
- **Decomposition**: Trend, seasonal, and residual components
- **ACF/PACF Analysis**: Autocorrelation plots
- **Multiple Forecasting Models**:
  - ARIMA
  - SARIMA (Seasonal ARIMA)
  - Exponential Smoothing
  - Prophet (Facebook's forecasting tool)
- **Model Comparison**: Automatic evaluation with metrics (RMSE, MAE, MAPE, R²)
- **Future Forecasting**: Predictions beyond your dataset
- **Residual Diagnostics**: Model validation
- **Report Generation**: Comprehensive text reports

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /Users/bobbymurphy/cs676/project2/deliverable2
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your Anthropic API key:**
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   ```

   Or add it to your `.bashrc` or `.zshrc`:
   ```bash
   echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.bashrc
   source ~/.bashrc
   ```

## Usage

### Running the Gradio App

```bash
python gradio_app.py
```

The app will launch at `http://localhost:7860`

### Using the Chatbot

Simply chat with Claude using natural language! Here are some example prompts:

**Getting Started:**
- "Generate sample data and run a complete analysis"
- "I've uploaded my sales data, can you analyze it?"

**Exploratory Analysis:**
- "Show me exploratory analysis of this data"
- "What are the summary statistics?"
- "Visualize the data distribution"

**Statistical Testing:**
- "Test if my data is stationary"
- "Is there significant autocorrelation?"
- "Show me ACF and PACF plots"

**Decomposition:**
- "Decompose the time series"
- "What's the trend in my data?"

**Modeling:**
- "Fit an ARIMA model"
- "Try ARIMA with p=2, d=1, q=2"
- "Build a SARIMA model with seasonality of 12"
- "Fit all available models"

**Evaluation:**
- "Compare all the models"
- "Which model performs best?"
- "Show me the forecast plots"
- "Analyze the residuals"

**Forecasting:**
- "Forecast the next 30 days"
- "Predict the next 90 days using the best model"

**Complete Analysis:**
- "Run a complete analysis on this data"
- "Do a full time series analysis"

### Using the Standalone Analyzer

You can also use the time series analyzer directly in Python:

```python
from time_series_analyzer import TimeSeriesAnalyzer

# Initialize analyzer
analyzer = TimeSeriesAnalyzer(freq='D')

# Option 1: Generate sample data
analyzer.generate_sample_data(n_periods=730, trend='linear', seasonality=True)

# Option 2: Load your own data
# analyzer.load_data('your_data.csv', date_column='date', value_column='value')

# Run analysis pipeline
(analyzer
    .explore_data()
    .test_stationarity()
    .plot_acf_pacf(lags=40)
    .decompose(model='additive')
    .split_data(test_size=0.2)
    .fit_arima(order=(2, 1, 2), name='ARIMA(2,1,2)')
    .fit_sarima(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), name='SARIMA')
    .fit_exponential_smoothing(seasonal='add', seasonal_periods=7)
    .fit_prophet()
    .compare_models()
    .plot_forecasts()
    .plot_residuals()
    .generate_report()
    .forecast_future(steps=60)
)
```

## File Structure

```
deliverable2/
├── gradio_app.py              # Gradio web interface with Claude chatbot
├── time_series_analyzer.py    # Core time series analysis class
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── main.py                     # Standalone script (same as time_series_analyzer.py)
```

## Output Files

When you run an analysis, the following files are generated:

### Plots (PNG images):
- `01_exploratory_analysis.png` - Data exploration and distributions
- `02_acf_pacf.png` - Autocorrelation plots
- `03_decomposition.png` - Trend/seasonal decomposition
- `04_forecasts.png` - Model forecast comparisons
- `05_residuals.png` - Residual diagnostics
- `06_future_forecast.png` - Future predictions with confidence intervals

### Data Files:
- `time_series_report.txt` - Comprehensive analysis report
- `future_forecast.csv` - Future predictions in CSV format

## Data Format

Your CSV file should have:
- A date/time column (will be auto-detected)
- A numeric value column (will be auto-detected)

Example:
```csv
date,value
2020-01-01,100.5
2020-01-02,102.3
2020-01-03,98.7
...
```

## Supported Frequencies

- `'D'` - Daily
- `'W'` - Weekly
- `'M'` - Monthly
- `'Q'` - Quarterly
- `'Y'` - Yearly
- `'H'` - Hourly
- And more (see pandas frequency strings)

## Model Parameters

### ARIMA(p, d, q)
- `p` - AR (autoregressive) order
- `d` - Differencing order
- `q` - MA (moving average) order

### SARIMA(p, d, q)(P, D, Q, s)
- `(p, d, q)` - Non-seasonal parameters
- `(P, D, Q, s)` - Seasonal parameters, where `s` is the seasonal period

### Exponential Smoothing
- `seasonal` - 'add' (additive) or 'mul' (multiplicative)
- `seasonal_periods` - Number of periods in a season

## Tips for Best Results

1. **Data Quality**: Ensure your data has no large gaps and is regularly spaced
2. **Stationarity**: Check stationarity before modeling; difference if needed
3. **Model Selection**: Try multiple models and compare performance
4. **Seasonality**: Use SARIMA or Prophet for data with strong seasonality
5. **Validation**: Always check residuals to validate model assumptions

## Troubleshooting

**"ANTHROPIC_API_KEY not set"**
- Make sure you've exported your API key in the terminal session

**Import errors**
- Run `pip install -r requirements.txt` again
- Ensure you're using Python 3.8 or higher

**Prophet installation issues**
- Prophet requires additional dependencies. Try:
  ```bash
  pip install pystan prophet
  ```

**Plot not displaying**
- Plots are saved as PNG files in the current directory
- Use the "Refresh Plots List" button in the Gradio interface

## Architecture

The application consists of two main components:

1. **TimeSeriesAnalyzer** (`time_series_analyzer.py`):
   - Core analysis engine
   - Implements all statistical tests and models
   - Handles data loading, processing, and visualization

2. **Gradio Frontend** (`gradio_app.py`):
   - Web interface for user interaction
   - Integrates Claude AI for natural language understanding
   - Tool-based architecture for executing analysis functions
   - Real-time plot viewing and file downloads

## License

MIT License

## Contributing

Feel free to submit issues or pull requests!

## Support

For questions or issues, please check:
- Time series analysis documentation: https://www.statsmodels.org/
- Gradio documentation: https://www.gradio.app/
- Anthropic Claude API: https://docs.anthropic.com/
