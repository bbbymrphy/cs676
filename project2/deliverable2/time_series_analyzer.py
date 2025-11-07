"""
Comprehensive Time Series Analysis Script
Performs complete time series analysis including:
- Data loading and exploration
- Preprocessing and cleaning
- Stationarity testing
- Decomposition
- Multiple modeling approaches (ARIMA, SARIMA, Prophet, Exponential Smoothing)
- Forecasting and evaluation
- Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Statistical tests
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

# Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

# ACF/PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

# Optional: Prophet (Facebook's forecasting tool)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Install with: pip install prophet")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)


class TimeSeriesAnalyzer:
    """Complete time series analysis pipeline"""

    def __init__(self, data=None, date_column=None, value_column=None, freq='D'):
        """
        Initialize the analyzer

        Parameters:
        -----------
        data : pd.DataFrame or str
            DataFrame or path to CSV file
        date_column : str
            Name of the date column
        value_column : str
            Name of the value column to analyze
        freq : str
            Frequency of time series ('D' for daily, 'M' for monthly, etc.)
        """
        self.freq = freq
        self.original_data = None
        self.data = None
        self.train = None
        self.test = None
        self.models = {}
        self.forecasts = {}
        self.metrics = {}

        if data is not None:
            self.load_data(data, date_column, value_column)

    def load_data(self, data, date_column=None, value_column=None):
        """Load and prepare time series data"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        if isinstance(data, str):
            self.original_data = pd.read_csv(data)
            print(f"Loaded data from: {data}")
        else:
            self.original_data = data.copy()

        print(f"Data shape: {self.original_data.shape}")
        print(f"\nColumns: {list(self.original_data.columns)}")
        print(f"\nFirst few rows:\n{self.original_data.head()}")

        # Prepare time series
        if date_column and value_column:
            df = self.original_data[[date_column, value_column]].copy()
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column)
            df = df.sort_index()
            self.data = df[value_column]
        else:
            # Assume first column is date, second is value
            df = self.original_data.copy()
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0])
            df = df.sort_index()
            self.data = df.iloc[:, 0]

        # Set frequency
        self.data = self.data.asfreq(self.freq)

        print(f"\nTime series prepared:")
        print(f"  - Start date: {self.data.index[0]}")
        print(f"  - End date: {self.data.index[-1]}")
        print(f"  - Length: {len(self.data)}")
        print(f"  - Frequency: {self.freq}")
        print(f"  - Missing values: {self.data.isna().sum()}")

        return self

    def generate_sample_data(self, n_periods=365, trend='linear', seasonality=True, noise_level=0.1):
        """Generate sample time series data for testing"""
        print("=" * 80)
        print("GENERATING SAMPLE DATA")
        print("=" * 80)

        dates = pd.date_range(start='2020-01-01', periods=n_periods, freq=self.freq)

        # Trend component
        if trend == 'linear':
            trend_component = np.linspace(100, 200, n_periods)
        elif trend == 'exponential':
            trend_component = 100 * np.exp(np.linspace(0, 0.7, n_periods))
        else:
            trend_component = np.ones(n_periods) * 100

        # Seasonal component
        if seasonality:
            seasonal_component = 20 * np.sin(2 * np.pi * np.arange(n_periods) / 365)
        else:
            seasonal_component = np.zeros(n_periods)

        # Noise
        noise = np.random.normal(0, noise_level * np.mean(trend_component), n_periods)

        # Combine
        values = trend_component + seasonal_component + noise

        self.data = pd.Series(values, index=dates, name='value')
        self.data = self.data.asfreq(self.freq)

        print(f"Generated {n_periods} data points")
        print(f"  - Trend: {trend}")
        print(f"  - Seasonality: {seasonality}")
        print(f"  - Noise level: {noise_level}")

        return self

    def explore_data(self):
        """Explore and visualize the time series"""
        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 80)

        # Summary statistics
        print("\nSummary Statistics:")
        print(self.data.describe())

        # Plot time series
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))

        # Original series
        axes[0].plot(self.data.index, self.data.values, linewidth=1)
        axes[0].set_title('Time Series Plot', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)

        # Distribution
        axes[1].hist(self.data.values, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_title('Distribution of Values', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)

        # Box plot by period (if applicable)
        if len(self.data) > 365:
            data_with_year = pd.DataFrame({
                'value': self.data.values,
                'year': self.data.index.year
            })
            axes[2].boxplot([data_with_year[data_with_year['year'] == year]['value'].values
                            for year in sorted(data_with_year['year'].unique())],
                           labels=sorted(data_with_year['year'].unique()))
            axes[2].set_title('Distribution by Year', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Year')
            axes[2].set_ylabel('Value')
        else:
            axes[2].text(0.5, 0.5, 'Not enough data for yearly comparison',
                        ha='center', va='center', transform=axes[2].transAxes)

        plt.tight_layout()
        plt.savefig('01_exploratory_analysis.png', dpi=300, bbox_inches='tight')
        print("\nSaved plot: 01_exploratory_analysis.png")
        plt.show()

        return self

    def test_stationarity(self):
        """Test for stationarity using ADF and KPSS tests"""
        print("\n" + "=" * 80)
        print("STATIONARITY TESTING")
        print("=" * 80)

        # Remove any NaN values for testing
        data_clean = self.data.dropna()

        # Augmented Dickey-Fuller Test
        print("\n1. Augmented Dickey-Fuller Test:")
        print("   H0: Time series has a unit root (non-stationary)")
        print("   H1: Time series is stationary")

        adf_result = adfuller(data_clean, autolag='AIC')
        print(f"\n   ADF Statistic: {adf_result[0]:.4f}")
        print(f"   p-value: {adf_result[1]:.4f}")
        print(f"   Critical Values:")
        for key, value in adf_result[4].items():
            print(f"      {key}: {value:.4f}")

        if adf_result[1] < 0.05:
            print("\n    REJECT H0: Series is stationary (ADF test)")
        else:
            print("\n    FAIL TO REJECT H0: Series is non-stationary (ADF test)")

        # KPSS Test
        print("\n2. KPSS Test:")
        print("   H0: Time series is stationary")
        print("   H1: Time series has a unit root (non-stationary)")

        kpss_result = kpss(data_clean, regression='ct', nlags='auto')
        print(f"\n   KPSS Statistic: {kpss_result[0]:.4f}")
        print(f"   p-value: {kpss_result[1]:.4f}")
        print(f"   Critical Values:")
        for key, value in kpss_result[3].items():
            print(f"      {key}: {value:.4f}")

        if kpss_result[1] > 0.05:
            print("\n    FAIL TO REJECT H0: Series is stationary (KPSS test)")
        else:
            print("\n    REJECT H0: Series is non-stationary (KPSS test)")

        # Ljung-Box Test for autocorrelation
        print("\n3. Ljung-Box Test (Autocorrelation):")
        lb_result = acorr_ljungbox(data_clean, lags=[10], return_df=True)
        print(f"\n   Test Statistic: {lb_result['lb_stat'].values[0]:.4f}")
        print(f"   p-value: {lb_result['lb_pvalue'].values[0]:.4f}")

        if lb_result['lb_pvalue'].values[0] < 0.05:
            print("\n    Significant autocorrelation detected")
        else:
            print("\n    No significant autocorrelation")

        return self

    def plot_acf_pacf(self, lags=40):
        """Plot ACF and PACF"""
        print("\n" + "=" * 80)
        print("ACF AND PACF PLOTS")
        print("=" * 80)

        data_clean = self.data.dropna()

        fig, axes = plt.subplots(2, 1, figsize=(15, 8))

        plot_acf(data_clean, lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)', fontsize=14, fontweight='bold')

        plot_pacf(data_clean, lags=lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('02_acf_pacf.png', dpi=300, bbox_inches='tight')
        print("\nSaved plot: 02_acf_pacf.png")
        plt.show()

        return self

    def decompose(self, model='additive', period=None):
        """Decompose time series into trend, seasonal, and residual components"""
        print("\n" + "=" * 80)
        print("TIME SERIES DECOMPOSITION")
        print("=" * 80)

        data_clean = self.data.dropna()

        if period is None:
            # Try to infer period
            if self.freq == 'D':
                period = 365
            elif self.freq == 'M':
                period = 12
            elif self.freq == 'W':
                period = 52
            else:
                period = min(len(data_clean) // 2, 365)

        print(f"\nDecomposition model: {model}")
        print(f"Period: {period}")

        decomposition = seasonal_decompose(data_clean, model=model, period=period)

        fig, axes = plt.subplots(4, 1, figsize=(15, 12))

        decomposition.observed.plot(ax=axes[0])
        axes[0].set_ylabel('Observed')
        axes[0].set_title('Time Series Decomposition', fontsize=14, fontweight='bold')

        decomposition.trend.plot(ax=axes[1])
        axes[1].set_ylabel('Trend')

        decomposition.seasonal.plot(ax=axes[2])
        axes[2].set_ylabel('Seasonal')

        decomposition.resid.plot(ax=axes[3])
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')

        plt.tight_layout()
        plt.savefig('03_decomposition.png', dpi=300, bbox_inches='tight')
        print("\nSaved plot: 03_decomposition.png")
        plt.show()

        # Store decomposition
        self.decomposition = decomposition

        return self

    def split_data(self, test_size=0.2):
        """Split data into training and testing sets"""
        print("\n" + "=" * 80)
        print("TRAIN-TEST SPLIT")
        print("=" * 80)

        split_idx = int(len(self.data) * (1 - test_size))
        self.train = self.data[:split_idx]
        self.test = self.data[split_idx:]

        print(f"\nTrain size: {len(self.train)} ({(1-test_size)*100:.0f}%)")
        print(f"Test size: {len(self.test)} ({test_size*100:.0f}%)")
        print(f"\nTrain period: {self.train.index[0]} to {self.train.index[-1]}")
        print(f"Test period: {self.test.index[0]} to {self.test.index[-1]}")

        return self

    def fit_arima(self, order=(1, 1, 1), name='ARIMA'):
        """Fit ARIMA model"""
        print(f"\n" + "=" * 80)
        print(f"FITTING {name} MODEL")
        print("=" * 80)
        print(f"\nOrder (p, d, q): {order}")

        model = ARIMA(self.train, order=order)
        fitted_model = model.fit()

        print(f"\n{fitted_model.summary()}")

        # Store model
        self.models[name] = fitted_model

        # Forecast
        forecast = fitted_model.forecast(steps=len(self.test))
        self.forecasts[name] = forecast

        # Calculate metrics
        self._calculate_metrics(name, self.test, forecast)

        return self

    def fit_sarima(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), name='SARIMA'):
        """Fit SARIMA model"""
        print(f"\n" + "=" * 80)
        print(f"FITTING {name} MODEL")
        print("=" * 80)
        print(f"\nOrder (p, d, q): {order}")
        print(f"Seasonal order (P, D, Q, s): {seasonal_order}")

        model = SARIMAX(self.train, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(disp=False)

        print(f"\n{fitted_model.summary()}")

        # Store model
        self.models[name] = fitted_model

        # Forecast
        forecast = fitted_model.forecast(steps=len(self.test))
        self.forecasts[name] = forecast

        # Calculate metrics
        self._calculate_metrics(name, self.test, forecast)

        return self

    def fit_exponential_smoothing(self, seasonal='add', seasonal_periods=12, name='ExpSmoothing'):
        """Fit Exponential Smoothing model"""
        print(f"\n" + "=" * 80)
        print(f"FITTING {name} MODEL")
        print("=" * 80)
        print(f"\nSeasonal: {seasonal}")
        print(f"Seasonal periods: {seasonal_periods}")

        model = ExponentialSmoothing(
            self.train,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            trend='add'
        )
        fitted_model = model.fit()

        print(f"\nModel fitted successfully")

        # Store model
        self.models[name] = fitted_model

        # Forecast
        forecast = fitted_model.forecast(steps=len(self.test))
        self.forecasts[name] = forecast

        # Calculate metrics
        self._calculate_metrics(name, self.test, forecast)

        return self

    def fit_prophet(self, name='Prophet'):
        """Fit Prophet model"""
        if not PROPHET_AVAILABLE:
            print("\nProphet is not available. Skipping...")
            return self

        print(f"\n" + "=" * 80)
        print(f"FITTING {name} MODEL")
        print("=" * 80)

        # Prepare data for Prophet
        train_prophet = pd.DataFrame({
            'ds': self.train.index,
            'y': self.train.values
        })

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        model.fit(train_prophet)

        # Forecast
        future = pd.DataFrame({'ds': self.test.index})
        forecast = model.predict(future)

        # Store model and forecast
        self.models[name] = model
        self.forecasts[name] = pd.Series(
            forecast['yhat'].values,
            index=self.test.index
        )

        # Calculate metrics
        self._calculate_metrics(name, self.test, self.forecasts[name])

        print(f"\n{name} model fitted successfully")

        return self

    def _calculate_metrics(self, name, actual, predicted):
        """Calculate evaluation metrics"""
        mse = mean_squared_error(actual, predicted)
        rmse = sqrt(mse)
        mae = mean_absolute_error(actual, predicted)

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        # R-squared
        r2 = r2_score(actual, predicted)

        self.metrics[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }

    def compare_models(self):
        """Compare all fitted models"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)

        if not self.metrics:
            print("\nNo models have been fitted yet!")
            return self

        # Create comparison table
        comparison_df = pd.DataFrame(self.metrics).T
        comparison_df = comparison_df.round(4)
        comparison_df = comparison_df.sort_values('RMSE')

        print("\nMetrics Comparison (sorted by RMSE):")
        print(comparison_df)

        # Find best model
        best_model = comparison_df.index[0]
        print(f"\nBest Model: {best_model}")
        print(f"   RMSE: {comparison_df.loc[best_model, 'RMSE']:.4f}")
        print(f"   MAE: {comparison_df.loc[best_model, 'MAE']:.4f}")
        print(f"   MAPE: {comparison_df.loc[best_model, 'MAPE']:.2f}%")
        print(f"   R2: {comparison_df.loc[best_model, 'R2']:.4f}")

        return self

    def plot_forecasts(self, show_train=True):
        """Plot all forecasts"""
        print("\n" + "=" * 80)
        print("PLOTTING FORECASTS")
        print("=" * 80)

        if not self.forecasts:
            print("\nNo forecasts available!")
            return self

        plt.figure(figsize=(15, 8))

        # Plot training data
        if show_train:
            plt.plot(self.train.index, self.train.values,
                    label='Train', color='black', linewidth=2, alpha=0.7)

        # Plot test data
        plt.plot(self.test.index, self.test.values,
                label='Actual', color='blue', linewidth=2)

        # Plot forecasts
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, (name, forecast) in enumerate(self.forecasts.items()):
            color = colors[i % len(colors)]
            plt.plot(forecast.index, forecast.values,
                    label=f'{name} Forecast', color=color,
                    linewidth=2, linestyle='--', alpha=0.7)

        plt.title('Model Forecasts Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig('04_forecasts.png', dpi=300, bbox_inches='tight')
        print("\nSaved plot: 04_forecasts.png")
        plt.show()

        return self

    def plot_residuals(self, model_name=None):
        """Plot residual diagnostics"""
        print("\n" + "=" * 80)
        print("RESIDUAL DIAGNOSTICS")
        print("=" * 80)

        if model_name is None:
            # Use best model
            if self.metrics:
                comparison_df = pd.DataFrame(self.metrics).T
                model_name = comparison_df['RMSE'].idxmin()
            else:
                print("\nNo models available!")
                return self

        if model_name not in self.forecasts:
            print(f"\nModel '{model_name}' not found!")
            return self

        print(f"\nAnalyzing residuals for: {model_name}")

        # Calculate residuals
        residuals = self.test - self.forecasts[model_name]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Residuals over time
        axes[0, 0].plot(residuals.index, residuals.values)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_title('Residuals Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)

        # Histogram
        axes[0, 1].hist(residuals.values, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Residuals Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals.values, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # ACF of residuals
        plot_acf(residuals.dropna(), lags=20, ax=axes[1, 1])
        axes[1, 1].set_title('ACF of Residuals', fontweight='bold')

        plt.suptitle(f'Residual Diagnostics - {model_name}',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()

        plt.savefig('05_residuals.png', dpi=300, bbox_inches='tight')
        print("\nSaved plot: 05_residuals.png")
        plt.show()

        return self

    def generate_report(self, filename='time_series_report.txt'):
        """Generate a comprehensive analysis report"""
        print("\n" + "=" * 80)
        print("GENERATING REPORT")
        print("=" * 80)

        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TIME SERIES ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Data summary
            f.write("DATA SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total observations: {len(self.data)}\n")
            f.write(f"Start date: {self.data.index[0]}\n")
            f.write(f"End date: {self.data.index[-1]}\n")
            f.write(f"Frequency: {self.freq}\n")
            f.write(f"Missing values: {self.data.isna().sum()}\n\n")

            f.write(f"Descriptive Statistics:\n")
            f.write(f"{self.data.describe()}\n\n")

            # Model comparison
            if self.metrics:
                f.write("\nMODEL COMPARISON\n")
                f.write("-" * 80 + "\n")
                comparison_df = pd.DataFrame(self.metrics).T
                comparison_df = comparison_df.round(4)
                comparison_df = comparison_df.sort_values('RMSE')
                f.write(f"{comparison_df}\n\n")

                best_model = comparison_df.index[0]
                f.write(f"\nBest Model: {best_model}\n")
                f.write(f"  RMSE: {comparison_df.loc[best_model, 'RMSE']:.4f}\n")
                f.write(f"  MAE: {comparison_df.loc[best_model, 'MAE']:.4f}\n")
                f.write(f"  MAPE: {comparison_df.loc[best_model, 'MAPE']:.2f}%\n")
                f.write(f"  R2: {comparison_df.loc[best_model, 'R2']:.4f}\n\n")

        print(f"\nReport saved: {filename}")

        return self

    def forecast_future(self, steps=30, model_name=None):
        """Forecast future values"""
        print("\n" + "=" * 80)
        print("FUTURE FORECASTING")
        print("=" * 80)

        if model_name is None:
            # Use best model
            if self.metrics:
                comparison_df = pd.DataFrame(self.metrics).T
                model_name = comparison_df['RMSE'].idxmin()
            else:
                print("\nNo models available!")
                return self

        print(f"\nUsing model: {model_name}")
        print(f"Forecasting {steps} steps into the future")

        # Refit model on full data
        if 'ARIMA' in model_name:
            model = self.models[model_name]
            # Refit on full data
            full_model = ARIMA(self.data, order=model.specification['order'])
            fitted_full = full_model.fit()
            future_forecast = fitted_full.forecast(steps=steps)

        elif 'SARIMA' in model_name:
            model = self.models[model_name]
            full_model = SARIMAX(
                self.data,
                order=model.specification['order'],
                seasonal_order=model.specification['seasonal_order']
            )
            fitted_full = full_model.fit(disp=False)
            future_forecast = fitted_full.forecast(steps=steps)

        elif 'ExpSmoothing' in model_name:
            # Need to extract parameters and refit
            # For simplicity, just extend the forecast
            future_forecast = self.models[model_name].forecast(steps=steps)

        elif 'Prophet' in model_name and PROPHET_AVAILABLE:
            model = self.models[model_name]
            # Create future dates
            last_date = self.data.index[-1]
            future_dates = pd.date_range(start=last_date, periods=steps+1, freq=self.freq)[1:]
            future_df = pd.DataFrame({'ds': future_dates})
            forecast = model.predict(future_df)
            future_forecast = pd.Series(forecast['yhat'].values, index=future_dates)

        else:
            print(f"\nUnsupported model type: {model_name}")
            return self

        # Plot
        plt.figure(figsize=(15, 8))

        # Historical data
        plt.plot(self.data.index, self.data.values,
                label='Historical', color='black', linewidth=2)

        # Future forecast
        plt.plot(future_forecast.index, future_forecast.values,
                label='Forecast', color='red', linewidth=2, linestyle='--')

        # Confidence interval (approximate)
        std = self.data.std()
        plt.fill_between(future_forecast.index,
                        future_forecast.values - 1.96 * std,
                        future_forecast.values + 1.96 * std,
                        alpha=0.2, color='red', label='95% CI')

        plt.title(f'Future Forecast - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig('06_future_forecast.png', dpi=300, bbox_inches='tight')
        print("\nSaved plot: 06_future_forecast.png")
        plt.show()

        # Save forecast to CSV
        forecast_df = pd.DataFrame({
            'date': future_forecast.index,
            'forecast': future_forecast.values
        })
        forecast_df.to_csv('future_forecast.csv', index=False)
        print("Saved forecast: future_forecast.csv")

        return self


def main():
    """Main execution function with example usage"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TIME SERIES ANALYSIS")
    print("=" * 80)

    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer(freq='D')

    # Option 1: Generate sample data
    print("\nGenerating sample data for demonstration...")
    analyzer.generate_sample_data(n_periods=730, trend='linear', seasonality=True)

    # Option 2: Load your own data (uncomment and modify)
    # analyzer.load_data('your_data.csv', date_column='date', value_column='value')

    # Run complete analysis pipeline
    (analyzer
        .explore_data()
        .test_stationarity()
        .plot_acf_pacf(lags=40)
        .decompose(model='additive')
        .split_data(test_size=0.2)
        .fit_arima(order=(2, 1, 2), name='ARIMA(2,1,2)')
        .fit_sarima(order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), name='SARIMA')
        .fit_exponential_smoothing(seasonal='add', seasonal_periods=7, name='ExpSmoothing')
    )

    # Try Prophet if available
    if PROPHET_AVAILABLE:
        analyzer.fit_prophet(name='Prophet')

    # Compare models and visualize results
    (analyzer
        .compare_models()
        .plot_forecasts(show_train=True)
        .plot_residuals()
        .generate_report('time_series_report.txt')
        .forecast_future(steps=60)
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - 01_exploratory_analysis.png")
    print("  - 02_acf_pacf.png")
    print("  - 03_decomposition.png")
    print("  - 04_forecasts.png")
    print("  - 05_residuals.png")
    print("  - 06_future_forecast.png")
    print("  - time_series_report.txt")
    print("  - future_forecast.csv")


if __name__ == "__main__":
    main()
