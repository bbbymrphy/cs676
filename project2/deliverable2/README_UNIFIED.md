# Unified Machine Learning Analysis with Claude AI

An intelligent Gradio web application that combines **Time Series Analysis** and **Random Forest** machine learning with Claude AI's natural language understanding. Claude automatically selects the right tool for your task!

## 🎯 Key Features

### Intelligent Analysis Selection
Claude automatically determines whether to use:
- **Time Series Analysis** - For temporal forecasting and trend analysis
- **Random Forest** - For classification and regression tasks

### Natural Language Interface
Just describe what you want in plain English:
- "Forecast my sales for the next 30 days"
- "Build a model to predict customer churn"
- "What are the most important features in my data?"

## 📊 Capabilities

### Time Series Analysis
- Exploratory data analysis
- Stationarity testing (ADF, KPSS)
- Time series decomposition
- ACF/PACF analysis
- Multiple models: ARIMA, SARIMA, Exponential Smoothing, Prophet
- Automated forecasting
- Model comparison

### Random Forest Analysis
- Exploratory data analysis
- Data preprocessing (scaling, encoding, imputation)
- Classification and regression
- Feature importance analysis
- Cross-validation
- Hyperparameter tuning
- Model evaluation (confusion matrix, ROC curve, etc.)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /Users/bobbymurphy/cs676/project2/deliverable2
pip install -r requirements.txt
```

### 2. Set API Key

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

### 3. Run the App

```bash
python3 gradio_app_unified.py
```

Open your browser to: **http://localhost:7860**

## 💬 Example Conversations

### Time Series Examples

**You:** "Generate sample time series data and forecast the future"

**Claude:** Will:
1. Select time series analysis mode
2. Generate sample data with trend and seasonality
3. Perform complete analysis
4. Show forecasts with confidence intervals

---

**You:** "I have daily stock prices, help me forecast next month"

**Claude:** Will:
1. Load your data
2. Test for stationarity
3. Fit ARIMA/SARIMA models
4. Generate 30-day forecast

### Random Forest Examples

**You:** "Create a classification dataset and train a model"

**Claude:** Will:
1. Select random forest mode
2. Generate sample classification data
3. Train Random Forest classifier
4. Show accuracy, confusion matrix, ROC curve

---

**You:** "I want to predict house prices from features"

**Claude:** Will:
1. Detect regression task
2. Load your data
3. Preprocess features
4. Train Random Forest regressor
5. Show R², MAE, and residual plots

## 📁 File Structure

```
deliverable2/
├── gradio_app_unified.py          # Unified Gradio app (USE THIS!)
├── time_series_analyzer.py        # Time series engine
├── random_forest_analyzer.py      # Random forest engine
├── requirements.txt               # Dependencies
├── README_UNIFIED.md             # This file
└── [generated files]
    ├── 01_exploratory_analysis.png        # Time series plots
    ├── 02_acf_pacf.png
    ├── 03_decomposition.png
    ├── 04_forecasts.png
    ├── 05_residuals.png
    ├── 06_future_forecast.png
    ├── rf_01_exploratory_analysis.png     # Random forest plots
    ├── rf_02_evaluation.png
    ├── rf_03_feature_importance.png
    ├── time_series_report.txt
    └── random_forest_report.txt
```

## 🤖 How It Works

### 1. Analysis Type Selection

Claude uses a `select_analysis_type` tool to determine:

**Time Series** if request mentions:
- Forecasting, prediction over time
- Dates, timestamps, temporal
- ARIMA, seasonality, trends
- "next 30 days", "future values"

**Random Forest** if request mentions:
- Classification, regression
- Features, target variable
- Prediction from attributes
- Feature importance
- Churn, categories, labels

### 2. Tool Execution

Once the analysis type is selected, Claude uses specialized tools:

**Time Series Tools:**
- `ts_load_data` - Load temporal data
- `ts_generate_sample` - Create sample data
- `ts_explore` - Exploratory analysis
- `ts_test_stationarity` - Statistical tests
- `ts_decompose` - Decomposition
- `ts_fit_arima` - Fit ARIMA model
- `ts_forecast` - Generate forecasts
- `ts_complete_analysis` - Full pipeline

**Random Forest Tools:**
- `rf_load_data` - Load tabular data
- `rf_generate_sample` - Create sample data
- `rf_explore` - Exploratory analysis
- `rf_preprocess` - Data preprocessing
- `rf_train` - Train model
- `rf_evaluate` - Evaluate performance
- `rf_feature_importance` - Analyze features
- `rf_cross_validate` - Cross-validation
- `rf_complete_analysis` - Full pipeline

## 📈 Data Format

### Time Series Data
CSV with date and value columns:
```csv
date,value
2023-01-01,100
2023-01-02,102
2023-01-03,98
...
```

### Random Forest Data
CSV with features and target:
```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,0
2.3,4.5,6.7,1
...
```

## 🎓 Tips for Best Results

### For Time Series:
1. Ensure data is regularly spaced (no large gaps)
2. Include sufficient history (at least 50-100 observations)
3. Specify frequency ('D', 'M', 'W', etc.) if known
4. Check stationarity before modeling

### For Random Forest:
1. Include target column name if not auto-detected
2. More features generally help (but avoid multicollinearity)
3. Balance classification datasets if possible
4. Use cross-validation to check for overfitting

## 🔧 Troubleshooting

**"No module named 'seaborn'"**
```bash
pip install -r requirements.txt
```

**"ANTHROPIC_API_KEY not set"**
```bash
export ANTHROPIC_API_KEY='your-key'
# Or add to ~/.bashrc for persistence
```

**Claude chooses wrong analysis type**
Be explicit:
- "Use time series to forecast..."
- "Use random forest to classify..."

**Plots not showing**
- Click "Refresh Plots" button
- Check if files exist in current directory

## 🆚 When to Use Each Approach

### Use Time Series When:
- ✅ Data has timestamps/dates
- ✅ You want to forecast future values
- ✅ Looking for trends, seasonality, cycles
- ✅ One variable over time
- ✅ Examples: stock prices, sales over time, temperature

### Use Random Forest When:
- ✅ Predicting from multiple features
- ✅ Classification or regression task
- ✅ Want to understand feature importance
- ✅ No temporal component (or time is just a feature)
- ✅ Examples: spam detection, price prediction from features, churn prediction

## 💡 Advanced Usage

### Combining Both Approaches

You can use both in one session! Just ask Claude:

**You:** "First analyze my time series data, then use the forecasts as features in a random forest model"

Claude can:
1. Run time series analysis
2. Extract trend/seasonal components
3. Use these as features for classification/regression

### Custom Parameters

Be specific about what you want:

**Time Series:**
- "Fit ARIMA with p=2, d=1, q=2"
- "Forecast 90 days ahead"
- "Use multiplicative decomposition"

**Random Forest:**
- "Train with 200 trees and max depth of 15"
- "Show top 10 features"
- "Use 10-fold cross-validation"

## 📊 Output Files

### Plots
All generated as high-res PNG (300 DPI):
- Exploratory analysis
- Model evaluation
- Forecasts/predictions
- Feature importance
- Residual diagnostics

### Reports
Text files with:
- Data summary
- Model parameters
- Performance metrics
- Key findings

### Forecasts
CSV files with predictions for easy import to Excel/other tools

## 🌟 Example Use Cases

### Business Analytics
- Sales forecasting (Time Series)
- Customer segmentation (Random Forest)
- Inventory optimization (Time Series)
- Churn prediction (Random Forest)

### Finance
- Stock price forecasting (Time Series)
- Credit risk assessment (Random Forest)
- Portfolio optimization (Random Forest)

### Operations
- Demand forecasting (Time Series)
- Quality control classification (Random Forest)
- Anomaly detection (both)

## 🤝 Contributing

This is a comprehensive tool that can be extended with:
- Additional ML algorithms (XGBoost, Neural Networks)
- More preprocessing options
- Automated hyperparameter tuning
- Multi-variate time series
- Ensemble methods

## 📚 References

- **Time Series**: statsmodels, Prophet, ARIMA
- **Random Forest**: scikit-learn
- **Interface**: Gradio, Anthropic Claude
- **Visualization**: matplotlib, seaborn

## ⚡ Performance Notes

- Time series fitting can be slow for large datasets (>10,000 points)
- Random forest scales well with features but slower with samples
- Use sampling for initial exploration of very large datasets
- Prophet is slower than ARIMA but handles seasonality better

## 🔒 Security & Privacy

- All analysis runs locally
- No data sent to Claude except your text prompts
- API key only used for Claude conversation
- Generated files saved locally only

## 📖 License

MIT License - Feel free to use and modify!

---

**Ready to start?** Just run `python3 gradio_app_unified.py` and start chatting with Claude! 🚀
