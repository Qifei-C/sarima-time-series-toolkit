# Data Directory

This directory is for storing time series datasets for analysis. The datasets are not included in this repository to keep it lightweight.

## Data Sources

Download datasets from the sources listed in the main README.md file:

- **Economic Data**: FRED, World Bank, OECD
- **Energy Data**: EIA, Global Energy Observatory, Kaggle
- **Weather Data**: NOAA, European Climate Assessment
- **Financial Data**: Yahoo Finance, Alpha Vantage, Quandl
- **Transportation Data**: Government datasets
- **COVID-19 Data**: Johns Hopkins CSSE, Our World in Data

## Expected Format

Time series data should be in CSV format with:
- Date column (various formats supported)
- Value column(s) for the time series
- Optional metadata columns

Example:
```csv
date,value
2020-01-01,100.5
2020-01-02,102.3
2020-01-03,98.7
```

## Data Preprocessing

The toolkit includes data preprocessing utilities in `src/sarima_python.py` and `src/sarima_analysis.R` for:
- Date parsing and formatting
- Missing value handling
- Outlier detection
- Seasonal decomposition
- Stationarity testing