# EduPredict AI v3

> AI-Powered Academic Data Analysis Platform

## Overview

EduPredict AI v3 is an academic data analysis platform that provides:
- CSV data upload and processing
- Interactive visualizations
- ML forecasting with ARIMA models
- AI-powered Q&A system

## Features

- **Upload**: CSV file validation and preview
- **Process**: Handle missing values and transpose data
- **Visualize**: Interactive charts with PyGWalker
- **Predict**: ARIMA forecasting for academic metrics
- **Q&A**: Ask questions about your data

## Project Structure

```
v3_testing/
├── main.py                 # Main application
├── app/
│   ├── core/              # Core functionality
│   ├── ui/                # User interface
│   └── utils/             # Utilities
├── data/                  # Data storage
└── config/                # Configuration
```

## Quick Start

### Prerequisites
- Python 3.8+
- Google Gemini API Key

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API key in `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

3. Run the application:
```bash
streamlit run main.py
```

4. Open browser at `http://localhost:8501`

## Usage

1. **Upload**: Upload your CSV file and preview the data
2. **Process**: Handle missing values or transpose data if needed
3. **Visualize**: Create interactive charts and explore your data
4. **Predict**: Generate forecasts using ARIMA models
5. **Q&A**: Ask questions about your data and get AI-powered answers

## Core Components

- **Schema Mapper**: Uses Google Gemini to standardize CSV column structures
- **Data Handler**: Handles missing values using iterative imputation
- **Predictor**: ARIMA forecasting with automatic model selection
- **Visualizations**: Interactive charts with PyGWalker and Plotly

## Configuration

Create a `.env` file with your API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Troubleshooting

- **API Key Error**: Check your `.env` file
- **Import Error**: Run `pip install -r requirements.txt`
- **ARIMA Fails**: System automatically uses regression fallback

## License

MIT License