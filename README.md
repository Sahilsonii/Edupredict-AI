# 🎓 EduPredict AI v3 - Interactive Documentation

> **AI-Powered Academic Data Analysis Platform with Multi-Tab Interface**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [🏗️ Architecture](#️-architecture)
- [📊 Visual System Flow](#-visual-system-flow)
- [🎯 Features](#-features)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [💡 Usage Guide](#-usage-guide)
- [🔧 Core Components](#-core-components)
- [🤖 ML Pipeline](#-ml-pipeline)
- [📈 Visualization System](#-visualization-system)
- [⚙️ Configuration](#️-configuration)
- [🧪 Testing](#-testing)
- [🐛 Troubleshooting](#-troubleshooting)

---

## 🌟 Overview

**EduPredict AI v3** is a comprehensive academic data analysis platform that combines:
- **AI-Powered Schema Standardization** using Google Gemini
- **Advanced ML Forecasting** with ARIMA and RandomForest
- **Interactive Visualizations** with PyGWalker
- **Smart Data Processing** with iterative imputation
- **Multi-Tab Interface** for organized workflow

### Key Capabilities

```
┌─────────────────────────────────────────────────────────────┐
│                    EduPredict AI v3                         │
├─────────────────────────────────────────────────────────────┤
│  📁 Upload CSV  →  🔧 Process  →  📊 Visualize  →  🤖 Predict │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

### System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│                      (Streamlit Multi-Tab)                       │
├──────────────────────────────────────────────────────────────────┤
│  Tab 1: Upload  │  Tab 2: Process  │  Tab 3: Viz  │  Tab 4: ML  │
└────────┬─────────────────┬─────────────────┬──────────────┬──────┘
         │                 │                 │              │
         ▼                 ▼                 ▼              ▼
┌────────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐
│  Data Handler  │  │   Analyzer   │  │  Visual  │  │ Predictor│
│  - Imputation  │  │  - Structure │  │ PyGWalker│  │  - ARIMA │
│  - Transpose   │  │  - Context   │  │  Plotly  │  │  - RF    │
└────────┬───────┘  └──────┬───────┘  └────┬─────┘  └────┬─────┘
         │                 │                │              │
         └─────────────────┴────────────────┴──────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Schema Mapper   │
                    │  (LLM-Powered)   │
                    │  Google Gemini   │
                    └──────────────────┘
```

### Modular Structure

```
v3_testing/
│
├── main.py                    # 🎯 Entry Point (Multi-Tab Interface)
│
├── app/
│   ├── core/                  # 🧠 Business Logic
│   │   ├── analyzer.py        # CSV structure analysis
│   │   ├── data_handler.py    # Missing value imputation
│   │   ├── predictor.py       # ML forecasting engine
│   │   ├── schema_mapper.py   # LLM schema standardization
│   │   └── llm.py            # Gemini integration
│   │
│   ├── ui/                    # 🎨 User Interface
│   │   ├── dashboard.py       # Main dashboard logic
│   │   ├── sidebar.py         # Sidebar components
│   │   └── visualizations.py  # PyGWalker & Plotly
│   │
│   └── utils/                 # 🛠️ Utilities
│       └── helpers.py         # Educational keywords, regex
│
├── data/                      # 📊 Data Storage
│   ├── raw/                   # Uploaded CSV files
│   └── processed/             # Processed datasets
│
├── config/                    # ⚙️ Configuration
│   └── .env                   # API keys
│
└── tests/                     # 🧪 Unit Tests
```

---

## 📊 Visual System Flow

### Complete Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: DATA UPLOAD                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Upload CSV File │
                    │  - Validate      │
                    │  - Show Preview  │
                    └────────┬─────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  Smart Detection             │
              │  ✓ Missing Values?           │
              │  ✓ Needs Transpose?          │
              │  ✓ Unusual Structure?        │
              └──────────┬───────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 2: DATA PROCESSING                      │
└─────────────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │Transpose │   │ Impute   │   │ Cluster  │
   │  Data    │   │ Missing  │   │ Columns  │
   └──────────┘   └──────────┘   └──────────┘
         │               │               │
         └───────────────┴───────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 3: VISUALIZATION                        │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │   PyGWalker      │
              │   Interactive    │
              │   Dashboard      │
              └──────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 4: ML PREDICTIONS                       │
└─────────────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │ Schema   │   │  ARIMA   │   │Drill-Down│
   │Mapping   │   │Forecast  │   │Forecast  │
   └──────────┘   └──────────┘   └──────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 5: AI Q&A                               │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  Ask Questions   │
              │  Get AI Answers  │
              │  Download Report │
              └──────────────────┘
```

---

## 🎯 Features

### 1. 📁 Smart Data Upload (Tab 1)

```
Features:
├── Automatic file validation
├── Dataset preview (first 20 rows)
├── Smart suggestions:
│   ├── Missing value detection → "Go to Tab 2"
│   └── Transpose recommendation → "Go to Tab 2"
└── Metrics display (rows, columns, memory)
```

**Visual Example:**
```
┌─────────────────────────────────────────┐
│  📁 Upload CSV File                     │
├─────────────────────────────────────────┤
│  origin.csv (0.9MB)                     │
│  ✅ No missing values found!            │
│  💡 Column names are years → Transpose? │
├─────────────────────────────────────────┤
│  Rows: 20,411 | Columns: 5 | 5456 KB   │
└─────────────────────────────────────────┘
```

### 2. 🔧 Data Processing (Tab 2)

```
Processing Options:
├── Transpose Data
│   ├── Select index column
│   ├── Auto-rename duplicates
│   └── Smart year detection
│
└── Handle Missing Values
    ├── Basic Iterative Imputer
    │   ├── Max iterations: 1-20
    │   ├── Random state
    │   └── N nearest features
    │
    └── Advanced Iterative Imputer
        ├── Categorical encoding
        ├── Label encoding
        └── Type preservation
```

**Imputation Algorithm:**
```
Input: DataFrame with missing values
│
├─ Separate numeric & categorical columns
│
├─ Numeric Columns:
│  └─ IterativeImputer (sklearn)
│     ├─ Uses RandomForest internally
│     ├─ Predicts missing values
│     └─ Iterates until convergence
│
├─ Categorical Columns:
│  └─ Mode Imputation
│     └─ Fill with most frequent value
│
└─ Output: Complete DataFrame
```

### 3. 📊 Interactive Visualization (Tab 3)

```
PyGWalker Features:
├── Drag-and-drop interface
├── Chart types:
│   ├── Bar charts
│   ├── Line charts
│   ├── Scatter plots
│   ├── Pie charts
│   ├── Heatmaps
│   └── Area charts
│
├── Filters & Aggregations
└── Export capabilities
```

### 4. 🤖 ML Predictions (Tab 4)

```
ML Pipeline:
│
├── Schema Standardization (LLM)
│   ├── Detect domain (academic/non-academic)
│   ├── Map columns to roles:
│   │   ├── Metrics (enrollment, scores)
│   │   ├── Dimensions (department, gender)
│   │   └── Time (year, semester)
│   └── Reject non-academic data
│
├── Batch Forecasting
│   ├── ARIMA models (primary)
│   ├── Regression fallback
│   └── Confidence intervals
│
└── Drill-Down Forecasting
    ├── Select dimension (e.g., Department)
    ├── Select segment (e.g., "Engineering")
    ├── Select metric (e.g., "Students")
    └── Generate forecast with visualization
```

**ARIMA Forecasting Flow:**
```
Time Series Data
       │
       ▼
┌──────────────┐
│ Stationarity │
│    Test      │
│  (ADF Test)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Model Search │
│  Try p,d,q   │
│ combinations │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Select Best  │
│  (Lowest AIC)│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Forecast   │
│  Next Period │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Round Based  │
│  on Pattern  │
└──────────────┘
```

### 5. 💬 AI Q&A (Tab 5)

```
Q&A System:
├── Context Creation
│   ├── Analyze CSV structure
│   ├── Extract relevant data
│   └── Build smart context
│
├── FAISS Vector Store
│   ├── Embed CSV rows
│   ├── Semantic search
│   └── Retrieve relevant docs
│
└── Gemini LLM
    ├── Academic domain check
    ├── Answer generation
    └── Download report
```

---

## 📁 Project Structure

### File Descriptions

| File | Purpose | Key Functions |
|------|---------|---------------|
| `main.py` | Entry point with 5-tab interface | `main()`, `load_api_key()` |
| `analyzer.py` | CSV structure analysis | `analyze_csv_structure()`, `create_universal_context()` |
| `data_handler.py` | Missing value imputation | `iterative_impute()`, `advanced_iterative_impute()` |
| `predictor.py` | ML forecasting engine | `arima_forecast_students()`, `batch_forecast_backend()` |
| `schema_mapper.py` | LLM schema standardization | `SchemaMapper.standardize()` |
| `llm.py` | Gemini integration | `build_retriever()`, `get_answer_from_llm()` |
| `visualizations.py` | PyGWalker & Plotly | `show_interactive_pygwalker()` |
| `helpers.py` | Educational keywords | `EDUCATIONAL_KEYWORDS`, `extract_year_from_string()` |

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
Google Gemini API Key
```

### Installation

```bash
# 1. Clone repository
cd v3_testing

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
# Create .env file:
echo "GEMINI_API_KEY=your_api_key_here" > .env

# OR create secrets.json:
echo '{"GEMINI_API_KEY": "your_api_key_here"}' > secrets.json

# 4. Run application
streamlit run main.py
```

### First Run

```
1. Open browser at http://localhost:8501
2. Upload a CSV file (academic data recommended)
3. Follow smart suggestions
4. Explore tabs sequentially
5. Generate forecasts and ask questions
```

---

## 💡 Usage Guide

### Example Workflow

#### Scenario: Analyzing Student Enrollment Data

**Step 1: Upload** (Tab 1)
```
Upload: student_enrollment.csv
Columns: Year, Department, Students, Graduates
Rows: 150
✅ No missing values
💡 Suggestion: Data looks good!
```

**Step 2: Process** (Tab 2)
```
Skip (no missing values)
OR
Transpose if needed
```

**Step 3: Visualize** (Tab 3)
```
Drag "Year" to X-axis
Drag "Students" to Y-axis
Select "Line Chart"
Add "Department" to Color
→ See enrollment trends by department
```

**Step 4: Predict** (Tab 4)
```
AI Schema Mapping:
├── Time: Year
├── Metrics: Students, Graduates
└── Dimensions: Department

Batch Forecast Results:
├── students__by__Year: ✅ ARIMA(1,1,1)
└── graduates__by__Year: ✅ ARIMA(2,0,1)

Drill-Down:
├── Dimension: Department
├── Segment: "Engineering"
├── Metric: Students
└── Forecast: 1,245 students (2024)
```

**Step 5: Ask** (Tab 5)
```
Question: "What is the average enrollment in Engineering?"
Answer: "The average enrollment in the Engineering department
         is 1,180 students per year based on historical data
         from 2015-2023."
```

---

## 🔧 Core Components

### 1. Schema Mapper (LLM-Powered)

**Purpose:** Standardize diverse CSV structures into canonical schema

**Algorithm:**
```python
Input: Raw CSV with columns
│
├─ Extract column names + sample values
│
├─ Send to Gemini LLM with prompt:
│  "Map these columns to: metrics, dimensions, time"
│
├─ Receive JSON mapping
│
├─ Validate academic domain
│  ├─ If non-academic → Reject
│  └─ If academic → Continue
│
└─ Return standardized mapping
```

**Example:**
```
Raw CSV:
├── "Enrollment Count" → metric: "enrollment"
├── "Academic Year" → time: "year"
└── "Degree Type" → dimension: "degree_type"
```

### 2. Predictor Engine

**Clustering Algorithm:**
```
For each column:
│
├─ Year Detection:
│  ├─ Check column name (year, date, period)
│  ├─ Check values (YYYY pattern)
│  └─ Extract years → Create year column
│
├─ Numeric Clustering:
│  ├─ Use pd.qcut() for equal-sized bins
│  └─ Create binned column
│
└─ Categorical Clustering:
   ├─ Keep top 15 categories
   └─ Group others as "Other"
```

**Forecasting Algorithm:**
```
1. Aggregate data by time period
2. Test stationarity (ADF test)
3. Try ARIMA models (p,d,q combinations)
4. Select best model (lowest AIC)
5. Generate forecast
6. Round based on historical patterns
7. Calculate confidence intervals
```

### 3. Data Handler

**Iterative Imputation:**
```
Algorithm: MICE (Multiple Imputation by Chained Equations)

For iteration in 1..max_iter:
│
├─ For each column with missing values:
│  │
│  ├─ Use other columns as features
│  ├─ Train RandomForest model
│  ├─ Predict missing values
│  └─ Update column
│
└─ Repeat until convergence
```

---

## 🤖 ML Pipeline

### Complete ML Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: CSV FILE                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Schema Mapper (LLM) │
            │  - Domain validation │
            │  - Column mapping    │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Intelligent         │
            │  Clustering          │
            │  - Year extraction   │
            │  - Numeric binning   │
            │  - Category grouping │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Batch Forecasting   │
            │  - ARIMA models      │
            │  - Regression backup │
            │  - Confidence bounds │
            └──────────┬───────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT: PREDICTIONS                      │
│  - Next period forecasts                                    │
│  - Confidence intervals                                     │
│  - Model performance metrics                                │
│  - Drill-down capabilities                                  │
└─────────────────────────────────────────────────────────────┘
```

### Model Selection Logic

```
ARIMA Model Selection:
│
├─ For p in [0, 1, 2]:
│  └─ For q in [0, 1, 2]:
│     ├─ Fit ARIMA(p, d, q)
│     ├─ Calculate AIC
│     └─ Track best model
│
└─ Return model with lowest AIC

Fallback to Regression if ARIMA fails:
│
├─ Create lag features (lag_1, lag_2, lag_3)
├─ Train LinearRegression
└─ Generate forecast
```

---

## 📈 Visualization System

### PyGWalker Integration

```
Features:
├── Tableau-like interface
├── Drag-and-drop columns
├── Real-time chart updates
├── Multiple chart types
├── Filter capabilities
└── Export options
```

### Plotly Charts

```
Chart Types:
├── Time Series (Line + Markers)
├── Forecast Visualization
│   ├── Historical data (solid line)
│   ├── Forecast (dashed line)
│   └── Confidence interval (error bars)
├── Bar Charts (Interactive)
├── Scatter Plots
└── Heatmaps
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# .env file
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_API_KEY=your_google_api_key  # Alternative
```

### secrets.json (Alternative)

```json
{
  "GEMINI_API_KEY": "your_api_key_here",
  "GOOGLE_API_KEY": "your_api_key_here"
}
```

---

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_predictor.py

# With coverage
pytest --cov=app tests/
```

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| API Key Error | Check `.env` or `secrets.json` file |
| Import Error | Run `pip install -r requirements.txt` |
| ARIMA Fails | System automatically falls back to regression |
| PyGWalker Not Loading | Check `gw_config.json` exists |
| Non-Academic Data Rejected | Upload educational dataset |

---

## 📚 Educational Keywords Database

The system includes 250+ educational keywords across 10 categories:

```
Categories:
├── K-12 Subjects (35 keywords)
├── STEM Sciences (30 keywords)
├── Engineering (25 keywords)
├── Medical & Health (50 keywords)
├── Business & Economics (20 keywords)
├── Humanities (25 keywords)
├── Social Sciences (20 keywords)
├── Education & Teaching (15 keywords)
├── Professional Fields (20 keywords)
└── Metrics & Institutions (30 keywords)
```

---

## 🎓 Academic Domain Detection

```
Detection Algorithm:
│
├─ Check column names for keywords
├─ Check data values for keywords
├─ Calculate confidence score
│  └─ Score = (column_matches × 10) + (value_matches × 5)
│
└─ If score > 0 → Academic dataset
```

---

## 📊 Performance Metrics

```
System Performance:
├── CSV Upload: < 1 second
├── Schema Mapping: 2-5 seconds (LLM call)
├── Imputation: 5-30 seconds (depends on size)
├── ARIMA Forecast: 1-3 seconds per series
├── PyGWalker Load: 2-4 seconds
└── AI Q&A: 3-8 seconds (LLM call)
```

---

## 🔮 Future Enhancements

- [ ] Support for multiple file formats (Excel, JSON)
- [ ] Advanced ML models (LSTM, Prophet)
- [ ] Real-time data streaming
- [ ] Collaborative features
- [ ] Custom model training
- [ ] API endpoints
- [ ] Mobile app

---

## 📝 License

MIT License - See LICENSE file for details

---

## 👥 Contributors

Built with ❤️ by the EduPredict Team

---

## 📞 Support

For issues and questions:
- GitHub Issues
- Documentation
- Community Forum

---

**Made with Streamlit, Google Gemini, and lots of ☕**