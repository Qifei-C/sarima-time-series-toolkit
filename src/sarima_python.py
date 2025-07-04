#!/usr/bin/env python3
"""
Python implementation of SARIMA time series analysis
Based on the R implementation in draft4.Rmd
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class SARIMAAnalyzer:
    """
    A comprehensive SARIMA time series analyzer that mirrors the R implementation
    """
    
    def __init__(self, data_path=None, data=None, date_col=None, value_col=None):
        """
        Initialize the SARIMA analyzer
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file containing time series data
        data : pd.DataFrame, optional
            DataFrame containing the time series data
        date_col : str, optional
            Name of the date column
        value_col : str, optional
            Name of the value column
        """
        self.data_path = data_path
        self.raw_data = data
        self.date_col = date_col
        self.value_col = value_col
        self.ts_data = None
        self.train_data = None
        self.test_data = None
        self.model = None
        self.forecast = None
        
    def load_data(self):
        """Load and prepare the time series data"""
        if self.data_path:
            self.raw_data = pd.read_csv(self.data_path)
        
        if self.raw_data is None:
            raise ValueError("No data provided")
        
        # Handle different data formats
        if self.date_col and self.value_col:
            # Standard format with date and value columns
            self.raw_data[self.date_col] = pd.to_datetime(self.raw_data[self.date_col])
            self.raw_data = self.raw_data.sort_values(self.date_col)
            self.ts_data = self.raw_data.set_index(self.date_col)[self.value_col]
        else:
            # Assume first column is date, second is value
            self.ts_data = self.raw_data.iloc[:, 1]
            self.ts_data.index = pd.to_datetime(self.raw_data.iloc[:, 0])
            
        return self.ts_data
    
    def load_unemployment_data(self, file_path):
        """
        Load unemployment data in wide format (similar to R implementation)
        """
        df = pd.read_csv(file_path)
        
        # Reshape from wide to long format
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Melt the dataframe
        df_long = pd.melt(df, id_vars=['Year'], value_vars=months, 
                         var_name='Month', value_name='UnemploymentRate')
        
        # Create date column
        month_map = {month: f'{i+1:02d}' for i, month in enumerate(months)}
        df_long['Date'] = pd.to_datetime(df_long['Year'].astype(str) + '-' + 
                                        df_long['Month'].map(month_map) + '-01')
        
        # Sort by date
        df_long = df_long.sort_values('Date')
        
        # Create time series
        self.ts_data = df_long.set_index('Date')['UnemploymentRate']
        return self.ts_data
    
    def plot_timeseries(self, title="Time Series Plot"):
        """Plot the time series data"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.ts_data.index, self.ts_data.values)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def basic_statistics(self):
        """Calculate basic statistics and plots"""
        print("Basic Statistics:")
        print(f"Mean: {self.ts_data.mean():.4f}")
        print(f"Variance: {self.ts_data.var():.4f}")
        print(f"Standard Deviation: {self.ts_data.std():.4f}")
        print(f"Min: {self.ts_data.min():.4f}")
        print(f"Max: {self.ts_data.max():.4f}")
        
        # Plot with trend line
        plt.figure(figsize=(15, 10))
        
        # Time series plot with trend
        plt.subplot(2, 2, 1)
        plt.plot(self.ts_data.index, self.ts_data.values, alpha=0.7)
        plt.axhline(y=self.ts_data.mean(), color='red', linestyle='--', label='Mean')
        
        # Add trend line
        x_numeric = np.arange(len(self.ts_data))
        z = np.polyfit(x_numeric, self.ts_data.values, 1)
        p = np.poly1d(z)
        plt.plot(self.ts_data.index, p(x_numeric), color='blue', linewidth=2, label='Trend')
        
        plt.title('Time Series with Trend and Mean')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Histogram
        plt.subplot(2, 2, 2)
        plt.hist(self.ts_data.values, bins=40, density=True, alpha=0.7)
        plt.title('Time Series Histogram')
        plt.xlabel('Value')
        plt.ylabel('Density')
        
        # Box plot
        plt.subplot(2, 2, 3)
        plt.boxplot(self.ts_data.values)
        plt.title('Box Plot')
        plt.ylabel('Value')
        
        # Q-Q plot
        plt.subplot(2, 2, 4)
        stats.probplot(self.ts_data.values, dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        
        plt.tight_layout()
        plt.show()
    
    def box_cox_transformation(self):
        """Apply Box-Cox transformation"""
        from scipy.stats import boxcox
        
        # Ensure positive values for Box-Cox
        if self.ts_data.min() <= 0:
            shift = abs(self.ts_data.min()) + 1
            shifted_data = self.ts_data + shift
        else:
            shifted_data = self.ts_data
        
        # Apply Box-Cox transformation
        transformed_data, lambda_val = boxcox(shifted_data)
        
        print(f"Optimal lambda: {lambda_val:.4f}")
        
        # Plot comparison
        plt.figure(figsize=(15, 8))
        
        # Original data
        plt.subplot(2, 2, 1)
        plt.plot(self.ts_data.index, self.ts_data.values)
        plt.title('Original Data')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        # Transformed data
        plt.subplot(2, 2, 2)
        plt.plot(self.ts_data.index, transformed_data)
        plt.title(f'Box-Cox Transformed Data (λ={lambda_val:.4f})')
        plt.xlabel('Date')
        plt.ylabel('Transformed Value')
        plt.xticks(rotation=45)
        
        # Histogram comparison
        plt.subplot(2, 2, 3)
        plt.hist(self.ts_data.values, bins=40, density=True, alpha=0.7, label='Original')
        plt.title('Original Data Histogram')
        plt.xlabel('Value')
        plt.ylabel('Density')
        
        plt.subplot(2, 2, 4)
        plt.hist(transformed_data, bins=40, density=True, alpha=0.7, label='Transformed', color='orange')
        plt.title('Transformed Data Histogram')
        plt.xlabel('Value')
        plt.ylabel('Density')
        
        plt.tight_layout()
        plt.show()
        
        # Update the time series data
        self.ts_data = pd.Series(transformed_data, index=self.ts_data.index)
        
        print(f"Variance before transformation: {self.ts_data.var():.4f}")
        print(f"Variance after transformation: {pd.Series(transformed_data).var():.4f}")
        
        return lambda_val
    
    def train_test_split(self, test_size=0.25):
        """Split the data into training and testing sets"""
        split_idx = int(len(self.ts_data) * (1 - test_size))
        self.train_data = self.ts_data.iloc[:split_idx]
        self.test_data = self.ts_data.iloc[split_idx:]
        
        print(f"Training set size: {len(self.train_data)}")
        print(f"Test set size: {len(self.test_data)}")
        
        # Plot the split
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_data.index, self.train_data.values, label='Training', color='blue')
        plt.plot(self.test_data.index, self.test_data.values, label='Test', color='red')
        plt.title('Training and Test Sets')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def seasonal_decomposition(self):
        """Perform seasonal decomposition"""
        # Infer frequency if not set
        if len(self.train_data) > 24:
            period = 12  # Monthly data
        else:
            period = 4   # Quarterly data
        
        decomposition = seasonal_decompose(self.train_data, model='additive', period=period)
        
        # Plot decomposition
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        decomposition.observed.plot(ax=axes[0], title='Original')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition.resid.plot(ax=axes[3], title='Residual')
        
        plt.tight_layout()
        plt.show()
        
        return decomposition
    
    def check_stationarity(self, data=None):
        """Check stationarity using ADF test"""
        from statsmodels.tsa.stattools import adfuller
        
        if data is None:
            data = self.train_data
        
        result = adfuller(data.dropna())
        
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value}')
        
        if result[1] <= 0.05:
            print("Data is stationary")
        else:
            print("Data is non-stationary")
        
        return result[1] <= 0.05
    
    def difference_analysis(self):
        """Analyze different levels of differencing"""
        # First difference
        diff1 = self.train_data.diff().dropna()
        
        # Seasonal difference
        seasonal_diff = self.train_data.diff(12).dropna()
        
        # Both differences
        diff_both = self.train_data.diff().diff(12).dropna()
        
        # Plot differences
        plt.figure(figsize=(15, 12))
        
        plt.subplot(3, 2, 1)
        plt.plot(diff1.index, diff1.values)
        plt.title('First Difference')
        plt.xlabel('Date')
        plt.ylabel('Value')
        
        plt.subplot(3, 2, 2)
        plot_acf(diff1, lags=60, ax=plt.gca())
        plt.title('ACF - First Difference')
        
        plt.subplot(3, 2, 3)
        plt.plot(seasonal_diff.index, seasonal_diff.values)
        plt.title('Seasonal Difference (lag=12)')
        plt.xlabel('Date')
        plt.ylabel('Value')
        
        plt.subplot(3, 2, 4)
        plot_acf(seasonal_diff, lags=60, ax=plt.gca())
        plt.title('ACF - Seasonal Difference')
        
        plt.subplot(3, 2, 5)
        plt.plot(diff_both.index, diff_both.values)
        plt.title('Both Differences')
        plt.xlabel('Date')
        plt.ylabel('Value')
        
        plt.subplot(3, 2, 6)
        plot_acf(diff_both, lags=60, ax=plt.gca())
        plt.title('ACF - Both Differences')
        
        plt.tight_layout()
        plt.show()
        
        # Check stationarity
        print("Stationarity tests:")
        print("First difference:")
        self.check_stationarity(diff1)
        print("\nSeasonal difference:")
        self.check_stationarity(seasonal_diff)
        print("\nBoth differences:")
        self.check_stationarity(diff_both)
        
        return diff1, seasonal_diff, diff_both
    
    def acf_pacf_analysis(self, data=None, lags=60):
        """Plot ACF and PACF for model identification"""
        if data is None:
            data = self.train_data.diff().dropna()
        
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 2, 1)
        plot_acf(data, lags=lags, ax=plt.gca())
        plt.title('Autocorrelation Function (ACF)')
        
        plt.subplot(2, 2, 2)
        plot_pacf(data, lags=lags, ax=plt.gca())
        plt.title('Partial Autocorrelation Function (PACF)')
        
        plt.subplot(2, 2, 3)
        plt.plot(data.index, data.values)
        plt.title('Differenced Time Series')
        plt.xlabel('Date')
        plt.ylabel('Value')
        
        plt.subplot(2, 2, 4)
        plt.hist(data.values, bins=30, density=True, alpha=0.7)
        plt.title('Histogram of Differenced Data')
        plt.xlabel('Value')
        plt.ylabel('Density')
        
        plt.tight_layout()
        plt.show()
    
    def fit_sarima(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """Fit SARIMA model"""
        try:
            self.model = SARIMAX(self.train_data, 
                               order=order, 
                               seasonal_order=seasonal_order)
            self.model_fit = self.model.fit(disp=False)
            
            print("Model fitted successfully!")
            print(self.model_fit.summary())
            
            return self.model_fit
            
        except Exception as e:
            print(f"Error fitting model: {str(e)}")
            return None
    
    def model_diagnostics(self):
        """Perform model diagnostics"""
        if self.model_fit is None:
            print("No model fitted yet!")
            return
        
        # Residual analysis
        residuals = self.model_fit.resid
        
        plt.figure(figsize=(15, 10))
        
        # Residuals plot
        plt.subplot(2, 3, 1)
        plt.plot(residuals)
        plt.title('Residuals')
        plt.xlabel('Time')
        plt.ylabel('Residuals')
        
        # Histogram of residuals
        plt.subplot(2, 3, 2)
        plt.hist(residuals, bins=30, density=True, alpha=0.7)
        plt.title('Histogram of Residuals')
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        
        # Q-Q plot
        plt.subplot(2, 3, 3)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        
        # ACF of residuals
        plt.subplot(2, 3, 4)
        plot_acf(residuals, lags=40, ax=plt.gca())
        plt.title('ACF of Residuals')
        
        # PACF of residuals
        plt.subplot(2, 3, 5)
        plot_pacf(residuals, lags=40, ax=plt.gca())
        plt.title('PACF of Residuals')
        
        # Ljung-Box test
        plt.subplot(2, 3, 6)
        lb_stat, lb_pvalue = acorr_ljungbox(residuals, lags=10, return_df=True).iloc[-1]
        plt.text(0.1, 0.5, f'Ljung-Box Test:\nStatistic: {lb_stat:.4f}\np-value: {lb_pvalue:.4f}', 
                transform=plt.gca().transAxes, fontsize=12)
        plt.title('Ljung-Box Test')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Ljung-Box test p-value: {lb_pvalue:.4f}")
        if lb_pvalue > 0.05:
            print("Residuals appear to be white noise (good)")
        else:
            print("Residuals may not be white noise (consider model adjustment)")
    
    def forecast(self, steps=None):
        """Generate forecasts"""
        if self.model_fit is None:
            print("No model fitted yet!")
            return None
        
        if steps is None:
            steps = len(self.test_data)
        
        # Generate forecast
        forecast_result = self.model_fit.forecast(steps=steps)
        forecast_conf_int = self.model_fit.get_forecast(steps=steps).conf_int()
        
        # Create forecast index
        if len(self.test_data) > 0:
            forecast_index = self.test_data.index
        else:
            last_date = self.train_data.index[-1]
            forecast_index = pd.date_range(start=last_date, periods=steps+1, freq='M')[1:]
        
        # Create forecast series
        self.forecast = pd.Series(forecast_result, index=forecast_index)
        
        # Plot forecast
        plt.figure(figsize=(15, 8))
        
        # Plot training data
        plt.plot(self.train_data.index, self.train_data.values, label='Training', color='blue')
        
        # Plot test data if available
        if len(self.test_data) > 0:
            plt.plot(self.test_data.index, self.test_data.values, label='Actual', color='red')
        
        # Plot forecast
        plt.plot(forecast_index, forecast_result, label='Forecast', color='green')
        
        # Plot confidence interval
        plt.fill_between(forecast_index, 
                        forecast_conf_int.iloc[:, 0], 
                        forecast_conf_int.iloc[:, 1], 
                        alpha=0.3, color='green')
        
        plt.title('SARIMA Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return self.forecast
    
    def evaluate_forecast(self):
        """Evaluate forecast performance"""
        if self.forecast is None or len(self.test_data) == 0:
            print("No forecast or test data available!")
            return None
        
        # Calculate metrics
        mse = mean_squared_error(self.test_data, self.forecast)
        mae = mean_absolute_error(self.test_data, self.forecast)
        rmse = np.sqrt(mse)
        
        # Calculate percentage errors
        mape = np.mean(np.abs((self.test_data - self.forecast) / self.test_data)) * 100
        
        print("Forecast Evaluation Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    def grid_search_sarima(self, p_values, d_values, q_values, 
                          P_values, D_values, Q_values, s=12):
        """Grid search for optimal SARIMA parameters"""
        best_aic = float('inf')
        best_params = None
        results = []
        
        total_models = len(p_values) * len(d_values) * len(q_values) * len(P_values) * len(D_values) * len(Q_values)
        current_model = 0
        
        print(f"Testing {total_models} models...")
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    for P in P_values:
                        for D in D_values:
                            for Q in Q_values:
                                current_model += 1
                                try:
                                    order = (p, d, q)
                                    seasonal_order = (P, D, Q, s)
                                    
                                    model = SARIMAX(self.train_data, 
                                                  order=order, 
                                                  seasonal_order=seasonal_order)
                                    model_fit = model.fit(disp=False)
                                    
                                    aic = model_fit.aic
                                    results.append({
                                        'order': order,
                                        'seasonal_order': seasonal_order,
                                        'AIC': aic
                                    })
                                    
                                    if aic < best_aic:
                                        best_aic = aic
                                        best_params = (order, seasonal_order)
                                    
                                    if current_model % 10 == 0:
                                        print(f"Progress: {current_model}/{total_models}")
                                        
                                except Exception as e:
                                    continue
        
        print(f"\nBest model: SARIMA{best_params[0]} x {best_params[1]}")
        print(f"Best AIC: {best_aic:.4f}")
        
        # Fit the best model
        self.fit_sarima(order=best_params[0], seasonal_order=best_params[1])
        
        return best_params, results

# Example usage
if __name__ == "__main__":
    # Example with unemployment data
    analyzer = SARIMAAnalyzer()
    
    # Load unemployment data (adjust path as needed)
    try:
        ts_data = analyzer.load_unemployment_data("data/USUnemployment.csv")
        print(f"Loaded {len(ts_data)} observations")
        
        # Basic analysis
        analyzer.plot_timeseries("US Unemployment Rate")
        analyzer.basic_statistics()
        
        # Train-test split
        analyzer.train_test_split(test_size=0.25)
        
        # Stationarity and differencing
        analyzer.check_stationarity()
        analyzer.difference_analysis()
        
        # ACF/PACF analysis
        analyzer.acf_pacf_analysis()
        
        # Fit SARIMA model
        analyzer.fit_sarima(order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
        
        # Model diagnostics
        analyzer.model_diagnostics()
        
        # Forecast
        forecast = analyzer.forecast()
        
        # Evaluate
        metrics = analyzer.evaluate_forecast()
        
    except FileNotFoundError:
        print("Data file not found. Please check the path.")
    except Exception as e:
        print(f"Error: {str(e)}")