# SARIMA Time Series Analysis Toolkit

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![R 4.0+](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comprehensive toolkit for SARIMA (Seasonal AutoRegressive Integrated Moving Average) time series analysis and forecasting. Supports both Python and R implementations with advanced statistical methods for economic forecasting, demand prediction, and trend analysis.

## 🎯 Features

- **Complete SARIMA Implementation**: (p,d,q)(P,D,Q)s modeling
- **Dual Language Support**: Both Python and R implementations
- **Automatic Model Selection**: Grid search and information criteria
- **Comprehensive Diagnostics**: Residual analysis and model validation
- **Forecasting Tools**: Point and interval predictions
- **Visualization Suite**: Professional time series plots

## 🚀 Quick Start

### Python Implementation
```python
from src.sarima_python import SARIMAModel

# Load and analyze time series (download data from sources above)
model = SARIMAModel()
model.load_data('electricity_production.csv', date_col='date', value_col='production')

# Automatic model selection
best_model = model.auto_sarima()

# Generate forecasts
forecasts = model.forecast(periods=24, confidence_intervals=True)

# Visualize results
model.plot_forecast(forecasts, save_path='forecasts.png')
```

### R Implementation
```r
source('src/sarima_analysis.R')

# Load time series data (download from sources above)
ts_data <- load_time_series('electricity_production.csv')

# Fit SARIMA model
sarima_model <- fit_sarima(ts_data, order=c(1,1,1), seasonal=c(1,1,1))

# Generate forecasts
forecasts <- forecast_sarima(sarima_model, h=24)

# Create visualizations
plot_sarima_results(forecasts)
```

## 📊 Model Selection

### Information Criteria
- **AIC**: Akaike Information Criterion
- **BIC**: Bayesian Information Criterion  
- **AICc**: Corrected AIC for small samples
- **HQIC**: Hannan-Quinn Information Criterion

### Diagnostic Tests
- **Ljung-Box Test**: Residual autocorrelation
- **Jarque-Bera Test**: Normality of residuals
- **ARCH Test**: Heteroscedasticity detection
- **Stability Tests**: Parameter stability analysis

## 📁 Project Structure

```
sarima-time-series-toolkit/
├── src/                        # Source code
│   ├── sarima_python.py        # Python implementation
│   ├── sarima_analysis.R       # R implementation
│   └── diagnostics.py          # Diagnostic tools
├── examples/                   # Example notebooks
│   ├── electricity_analysis.py # Energy forecasting example
│   └── economic_indicators.R   # Economic time series
├── tests/                      # Test suite
├── docs/                       # Documentation
├── notebooks/                  # Jupyter/R notebooks
└── README.md                  # This file
```

## 📊 Data Sources

This toolkit works with various time series datasets. You can download data from:

**Economic Data:**
- [FRED Economic Data](https://fred.stlouisfed.org/) - Federal Reserve Economic Data
- [World Bank Open Data](https://data.worldbank.org/) - Global economic indicators
- [OECD Statistics](https://stats.oecd.org/) - Economic statistics

**Energy & Utilities:**
- [EIA Open Data](https://www.eia.gov/opendata/) - U.S. Energy Information Administration
- [Global Energy Observatory](http://globalenergyobservatory.org/) - Global energy data
- [Turbine Data](https://www.kaggle.com/datasets) - Wind turbine production data

**Weather & Climate:**
- [NOAA Climate Data](https://www.ncdc.noaa.gov/data-access) - Climate time series
- [European Climate Assessment](https://www.ecad.eu/) - European weather data

**Financial Data:**
- [Yahoo Finance](https://finance.yahoo.com/) - Stock prices and indices
- [Alpha Vantage](https://www.alphavantage.co/) - Financial market data
- [Quandl](https://www.quandl.com/) - Financial and economic data

**Transportation:**
- [Traffic Data](https://data.gov/) - Government traffic datasets
- [Transport Statistics](https://www.gov.uk/government/collections/transport-statistics-great-britain) - UK transport data

**COVID-19 Data:**
- [Johns Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19) - Global COVID-19 data
- [Our World in Data](https://ourworldindata.org/coronavirus) - COVID-19 statistics

## 🔧 Advanced Features

### Seasonal Decomposition
```python
# STL decomposition
decomposition = model.seasonal_decompose(method='stl')
model.plot_decomposition(decomposition)

# X-13ARIMA-SEATS (if available)
x13_decomp = model.x13_decompose()
```

### Model Diagnostics
```python
# Comprehensive residual analysis
diagnostics = model.residual_diagnostics()
print(f"Ljung-Box p-value: {diagnostics['ljung_box_pvalue']}")
print(f"Normality test p-value: {diagnostics['normality_pvalue']}")

# Plot diagnostic charts
model.plot_diagnostics(save_path='diagnostics/')
```

### Multiple Series Analysis
```python
# Vector ARIMA for multiple related series
from src.vector_arima import VARIMAModel

var_model = VARIMAModel()
var_model.fit_multiple_series([series1, series2, series3])
joint_forecasts = var_model.forecast(periods=12)
```

## 📈 Applications

### Economic Forecasting
- GDP growth prediction
- Inflation rate modeling
- Unemployment forecasting
- Stock price analysis

### Energy & Utilities
- Electricity demand forecasting
- Natural gas consumption
- Renewable energy production
- Peak load prediction

### Business Analytics
- Sales forecasting
- Inventory planning
- Demand prediction
- Revenue modeling

## 🏆 Performance Benchmarks

Model performance will depend on your specific time series characteristics, data quality, and parameter tuning. SARIMA models are designed to capture seasonal patterns and trends effectively when properly configured.

## 🔬 Research Applications

This toolkit supports research in:
- Economic time series analysis
- Energy demand forecasting
- Climate data modeling
- Financial econometrics

## 🤝 Contributing

We welcome contributions! Please ensure compatibility with both Python and R implementations.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📚 References

```bibtex
@book{box2015time,
  title={Time series analysis: forecasting and control},
  author={Box, George EP and Jenkins, Gwilym M and Reinsel, Gregory C and Ljung, Greta M},
  year={2015},
  publisher={John Wiley \& Sons}
}
```

---

📈 **Professional Time Series Forecasting**