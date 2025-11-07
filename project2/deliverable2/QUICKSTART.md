# Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Step 1: Install Dependencies

```bash
cd /Users/bobbymurphy/cs676/project2/deliverable2

# Run the setup script
./setup.sh

# Or manually:
pip install -r requirements.txt
```

### Step 2: Set Your API Key

You need an Anthropic API key to use Claude. Get one at: https://console.anthropic.com/

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

### Step 3: Launch the App

```bash
python gradio_app.py
```

Open your browser to: **http://localhost:7860**

---

## 💬 Example Conversations

### First Time Use (No Data)

**You:** "Generate sample data and run a complete analysis"

**Claude:** Will generate sample time series data and perform a full analysis including:
- Exploratory data analysis
- Stationarity testing
- ACF/PACF plots
- Decomposition
- Multiple models (ARIMA, SARIMA, Exponential Smoothing, Prophet)
- Model comparison
- Forecasting

### With Your Own Data

1. Click "Upload CSV File"
2. Upload your time series data (must have a date column and value column)
3. Chat with Claude:

**You:** "I've uploaded my sales data. Can you analyze it?"

**Claude:** Will automatically detect your columns and perform analysis

---

## 📊 Common Tasks

### Exploratory Analysis
- "Show me exploratory analysis"
- "What are the summary statistics?"
- "Plot the data distribution"

### Check Stationarity
- "Is my data stationary?"
- "Test for stationarity and explain the results"

### Build Models
- "Fit an ARIMA model"
- "Build a SARIMA model with 12-month seasonality"
- "Fit all available forecasting models"

### Compare Models
- "Which model is best?"
- "Compare all the models"
- "Show me model performance metrics"

### Forecast Future
- "Forecast the next 30 days"
- "Predict the next quarter using the best model"

### Get Results
- "Generate a report"
- "Show me all the plots"

---

## 📁 Your Data Format

CSV file with date and value columns:

```csv
date,sales
2023-01-01,1000
2023-01-02,1050
2023-01-03,980
...
```

Column names can vary - Claude will auto-detect them!

---

## 🎯 Pro Tips

1. **Start Simple**: Try "Generate sample data and run complete analysis" first
2. **Be Specific**: "Forecast next 60 days" is better than "forecast"
3. **Ask Questions**: "What does this mean?" or "Why is this important?"
4. **View Plots**: Use the plot selector on the right to view generated visualizations
5. **Download Results**: Reports and forecasts are downloadable from the right panel

---

## 🐛 Troubleshooting

**API Key Issues:**
```bash
# Check if set
echo $ANTHROPIC_API_KEY

# Set it for current session
export ANTHROPIC_API_KEY='sk-ant-...'

# Set it permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
```

**Import Errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# For Prophet issues specifically:
pip install pystan prophet
```

**Port Already in Use:**
```bash
# The app runs on port 7860 by default
# If that's taken, edit gradio_app.py and change:
# server_port=7860  →  server_port=7861
```

---

## 📚 Learn More

- Full documentation: See [README.md](README.md)
- Time series analysis: Check `time_series_analyzer.py`
- Example usage: See examples in the Gradio interface

---

## 🎉 You're Ready!

Just run `python gradio_app.py` and start chatting with Claude about your time series data!
