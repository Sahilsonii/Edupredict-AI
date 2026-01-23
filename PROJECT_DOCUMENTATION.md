# 📚 EduPredict v2 - Complete Project Documentation

## 🎯 Project Overview

### What is EduPredict?
EduPredict is a **Universal CSV Analysis Platform** powered by AI that can analyze ANY type of CSV dataset - from educational data (students, enrollments) to sales, financial, scientific, or any other structured data.

### Who is it for?
- **Data Analysts** who need quick insights from CSV files
- **Educational Institutions** tracking student performance and enrollment trends
- **Business Analysts** analyzing sales, inventory, or customer data
- **Researchers** working with experimental or survey data
- **Anyone** who wants to understand their CSV data without coding

### Key Capabilities
1. **🤖 AI-Powered Q&A**: Ask questions in natural language using Google Gemini AI
2. **🔮 ML Predictions**: Automatic machine learning predictions and forecasting
3. **📊 Interactive Visualizations**: Power BI-style charts with PyGwalker
4. **🛠️ Data Preprocessing**: Handle missing values, transpose data, cluster columns
5. **📈 Time-Series Forecasting**: ARIMA-based predictions with fallback regression
6. **🎯 Auto-Detection**: Automatically understands your data structure

---

## 📁 Project Structure

```
EduPredictv2/
├── app.py                      # Main Streamlit application (UI + orchestration)
├── predictions.py              # ML predictions & forecasting engine
├── missing_value_handler.py    # Data cleaning & imputation
├── visualization.py            # Interactive charts & visualizations
├── main.py                     # Entry point (minimal)
├── secrets.json               # API keys (GEMINI_API_KEY)
├── gw_config.json             # PyGwalker configuration
├── uploaded_data.csv          # Temporary storage for uploaded files
├── pyproject.toml             # Project dependencies
├── uv.lock                    # Dependency lock file
└── README.md                  # Project readme

```

### Ideal File Structure (Best Practices)
```
EduPredictv2/
├── src/                       # Source code
│   ├── __init__.py
│   ├── app.py                 # Main application
│   ├── core/                  # Core functionality
│   │   ├── __init__.py
│   │   ├── predictions.py
│   │   ├── preprocessing.py
│   │   └── ai_engine.py
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── missing_values.py
│   │   └── validators.py
│   └── visualization/         # Visualization modules
│       ├── __init__.py
│       └── charts.py
├── tests/                     # Unit tests
│   ├── test_predictions.py
│   └── test_preprocessing.py
├── data/                      # Data storage
│   ├── uploads/
│   └── cache/
├── config/                    # Configuration files
│   ├── secrets.json
│   └── settings.yaml
├── docs/                      # Documentation
│   └── API.md
├── requirements.txt           # Dependencies
├── .env                       # Environment variables
├── .gitignore
└── README.md
```

---

## 🔧 File-by-File Breakdown


### 1️⃣ app.py - Main Application (Heart of the System)

**Purpose**: Orchestrates the entire application - UI, data processing, AI integration, and user interactions.

#### Key Components:

##### A. Configuration & Setup (Lines 1-35)
```python
import streamlit as st
import pandas as pd
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
```

**What it does:**
- Imports all necessary libraries
- Sets up async event loop for Streamlit (required for LangChain)
- Configures page layout with `st.set_page_config()`

**Concepts Used:**
- **Streamlit**: Web framework for data apps (alternative: Flask, FastAPI, Dash)
- **Asyncio**: Handles asynchronous operations (alternative: threading, multiprocessing)
- **LangChain**: Framework for LLM applications (alternative: direct API calls, LlamaIndex)

**Why these choices?**
- Streamlit: Fastest way to build data apps without HTML/CSS/JS
- LangChain: Simplifies RAG (Retrieval Augmented Generation) implementation
- Asyncio: Required by Streamlit's threading model

---

##### B. API Key Management (Lines 37-44)
```python
@st.cache_data
def load_api_key():
    secrets_file = Path("secrets.json")
    if secrets_file.exists():
        with open(secrets_file, 'r') as f:
            secrets = json.load(f)
            return secrets.get('GEMINI_API_KEY')
    return None
```

**Line-by-line explanation:**
- `@st.cache_data`: Caches function result to avoid re-reading file on every rerun
- `Path("secrets.json")`: Creates path object (safer than string paths)
- `secrets_file.exists()`: Checks if file exists before opening
- `json.load(f)`: Parses JSON file into Python dictionary
- `secrets.get('GEMINI_API_KEY')`: Safely retrieves key (returns None if missing)

**Concepts:**
- **Caching**: Stores computed results to improve performance
- **Defensive Programming**: Check file exists before reading
- **JSON**: Standard format for configuration files

**Alternatives:**
- Environment variables: `os.getenv('GEMINI_API_KEY')`
- .env files with python-dotenv: `load_dotenv(); os.getenv('KEY')`
- Cloud secret managers: AWS Secrets Manager, Google Secret Manager

**Potential Errors:**
- ❌ FileNotFoundError: If secrets.json doesn't exist
- ❌ JSONDecodeError: If secrets.json has invalid JSON syntax
- ❌ KeyError: If GEMINI_API_KEY is missing (handled with .get())

---

##### C. RAG System - Vector Store Builder (Lines 46-67)
```python
@st.cache_resource
def build_retriever(csv_path: str, api_key: str):
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Load CSV as documents
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()
    
    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    # Build FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever()
```

**What is RAG (Retrieval Augmented Generation)?**
RAG is a technique that enhances LLM responses by:
1. Converting your data into vector embeddings (numerical representations)
2. Storing embeddings in a vector database (FAISS)
3. When user asks a question, finding relevant data chunks
4. Sending relevant data + question to LLM for accurate answers

**Line-by-line breakdown:**

1. `@st.cache_resource`: Caches the retriever object (different from cache_data - for non-serializable objects)

2. `CSVLoader(file_path=csv_path)`: 
   - Loads CSV and converts each row into a "Document" object
   - Each document has content (row data) and metadata (row number, column names)

3. `docs = loader.load()`:
   - Returns list of Document objects
   - Example: [Document(page_content="Name: John, Age: 25", metadata={"row": 1}), ...]

4. `GoogleGenerativeAIEmbeddings(model="models/embedding-001")`:
   - Creates embedding model that converts text to 768-dimensional vectors
   - Embeddings capture semantic meaning (similar text = similar vectors)

5. `FAISS.from_documents(docs, embeddings)`:
   - FAISS = Facebook AI Similarity Search
   - Creates vector database for fast similarity search
   - Stores all document embeddings in memory

6. `vectorstore.as_retriever()`:
   - Converts vector store to retriever interface
   - Retriever can find top-k most relevant documents for a query

**Concepts:**
- **Vector Embeddings**: Numerical representation of text (alternative: TF-IDF, Word2Vec)
- **Vector Database**: Stores and searches embeddings (alternatives: Pinecone, Weaviate, Chroma)
- **Semantic Search**: Find similar meaning, not just keywords

**Alternatives to FAISS:**
- Chroma: Persistent storage, easier to use
- Pinecone: Cloud-based, scalable
- Weaviate: GraphQL API, production-ready
- Simple approach: Use pandas filtering (no embeddings)

**Potential Errors:**
- ❌ API key invalid: Check GEMINI_API_KEY
- ❌ CSV too large: FAISS loads everything in memory
- ❌ Rate limiting: Too many embedding requests

---

##### D. CSV Processing with ML Predictions (Lines 69-150)
```python
@st.cache_data
def process_csv_with_predictions(uploaded_file_name, df_csv):
    # Step 1: Intelligent clustering
    df_processed, cluster_mappings = predictions.cluster_all_columns(
        df_csv,
        numeric_bins=8,
        cat_top_k=15,
        treat_years=True,
        keep_original=True
    )
```

**What is Clustering?**
Clustering groups similar data points together to:
- Reduce dimensionality (8 bins instead of 1000 unique values)
- Improve ML model performance
- Make data easier to visualize

**Parameters explained:**
- `numeric_bins=8`: Group numeric values into 8 ranges (quartiles)
- `cat_top_k=15`: Keep top 15 categories, group rest as "Other"
- `treat_years=True`: Detect and extract year values (2023/24 → 2023)
- `keep_original=True`: Keep original columns + add clustered versions

**Example:**
```
Original: Age = [18, 19, 20, 21, 22, 23, 24, 25]
Clustered: Age__bin = ["18-20", "18-20", "18-20", "21-23", "21-23", "21-23", "24-25", "24-25"]
```


```python
    # Step 2: Detect columns
    time_cols = predictions.detect_time_columns(df_processed)
    numeric_cols = predictions.detect_numerical_columns(df_processed)
    
    # Step 3: Run batch predictions
    batch_results = predictions.batch_forecast_backend(
        df_processed,
        potential_time_cols=time_cols,
        target_cols=target_cols
    )
```

**What happens here:**
1. **Column Detection**: Automatically identifies year/date columns and numeric columns
2. **Batch Forecasting**: Tries to predict every numeric column using every time column
3. **ARIMA + Fallback**: Uses ARIMA first, falls back to regression if ARIMA fails

**Why batch processing?**
- User doesn't need to specify what to predict
- System tries all possible combinations
- Returns only successful predictions

**Potential Errors:**
- ❌ No time columns found: Dataset has no dates/years
- ❌ No numeric columns: Dataset is all text
- ❌ Insufficient data: Need at least 4 time periods for forecasting

---

### 2️⃣ predictions.py - ML Engine (Brain of the System)

**Purpose**: Handles all machine learning, forecasting, and data preprocessing.

#### Key Components:

##### A. Educational Keywords Database (Lines 25-130)
```python
EDUCATIONAL_KEYWORDS = {
    'k12_subjects': ['mathematics', 'math', 'algebra', ...],
    'engineering': ['chemical engineering', 'civil engineering', ...],
    'medical_health': ['medicine', 'nursing', 'cardiology', ...],
    # ... 250+ keywords across 10 categories
}
```

**Purpose**: Detect if dataset is educational and identify domain

**How it works:**
1. Scans column names for keywords
2. Samples data values for keywords
3. Returns confidence score and detected categories

**Use cases:**
- Customize analysis based on domain
- Provide domain-specific insights
- Suggest relevant visualizations

**Alternatives:**
- Machine learning classifier (train on labeled datasets)
- Rule-based system with regex patterns
- User manually selects domain

---

##### B. Column Type Detection (Lines 132-180)
```python
def auto_detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    year_cols = []
    numeric_cols = []
    categorical_cols = []
    educational_cols = []
    
    for col in df.columns:
        # Check for year columns
        if any(hint in col.lower() for hint in ["year", "date", "period"]):
            year_cols.append(col)
        # Check for numeric
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
```

**Concepts:**
- **Type Inference**: Automatically determine data types
- **Heuristics**: Use rules of thumb (column names, value patterns)
- **Pandas dtypes**: Built-in type system (int64, float64, object)

**Why not just use pandas dtypes?**
- Pandas can't detect "year" columns (they're just integers)
- Can't distinguish between categorical and free text
- Need domain-specific logic (educational keywords)

**Alternatives:**
- pandas.api.types.infer_dtype(): Basic type inference
- Great Expectations: Data validation library
- Manual user selection: Let user specify column types

---

##### C. Clustering Functions (Lines 182-280)

**1. Numeric Clustering (Binning)**
```python
def cluster_numeric_column(series: pd.Series, n_bins: int = 8) -> pd.Series:
    if s.nunique() > n_bins:
        bins = pd.qcut(s, q=n_bins, duplicates="drop")
    else:
        bins = pd.cut(s, bins=min(n_bins, s.nunique()))
```

**What is qcut vs cut?**
- `qcut`: Quantile-based (equal number of items per bin)
- `cut`: Range-based (equal width bins)

**Example:**
```
Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

qcut(q=3):
  Bin 1: [1, 2, 3]     # 3 items
  Bin 2: [4, 5, 6, 7]  # 4 items
  Bin 3: [8, 9, 10]    # 3 items

cut(bins=3):
  Bin 1: [1, 2, 3]     # Range 1-3.33
  Bin 2: [4, 5, 6]     # Range 3.33-6.66
  Bin 3: [7, 8, 9, 10] # Range 6.66-10
```

**Why use clustering?**
- Reduces cardinality (fewer unique values)
- Improves ML model performance
- Makes patterns more visible
- Handles outliers better

**Alternatives:**
- K-means clustering: More sophisticated
- Custom binning rules: Domain-specific ranges
- No clustering: Use raw values (can overfit)

---

**2. Categorical Clustering**
```python
def cluster_categorical_column(series: pd.Series, top_k: int = 15) -> pd.Series:
    top_categories = s.value_counts().nlargest(top_k).index.tolist()
    clustered = s.apply(lambda x: x if x in top_categories else "Other")
```

**Why group rare categories?**
- Reduces noise in ML models
- Prevents overfitting on rare values
- Improves visualization clarity

**Example:**
```
Original: ["USA", "UK", "Canada", "France", "Germany", "Italy", "Spain", "Portugal", ...]
Clustered: ["USA", "UK", "Canada", "France", "Germany", "Other", "Other", "Other", ...]
```

**Alternatives:**
- Frequency threshold: Group categories with <5% frequency
- Hierarchical grouping: Group by similarity (USA, Canada → North America)
- No grouping: Keep all categories (can cause issues with 1000+ categories)

---

##### D. ARIMA Forecasting (Lines 282-380)
```python
def arima_forecast_students(series: pd.Series) -> Dict[str, Any]:
    # Test for stationarity
    adf_test = adfuller(s)
    d = 1 if adf_test[1] > 0.05 else 0
    
    # Try different ARIMA models
    for p in range(3):
        for q in range(3):
            model = ARIMA(s, order=(p, d, q))
            results = model.fit()
```

**What is ARIMA?**
ARIMA = AutoRegressive Integrated Moving Average
- **AR (p)**: Uses past values to predict future (lag terms)
- **I (d)**: Differencing to make data stationary
- **MA (q)**: Uses past forecast errors

**ARIMA(p, d, q) parameters:**
- `p`: Number of lag observations (autoregressive terms)
- `d`: Degree of differencing (0 or 1 usually)
- `q`: Size of moving average window

**Example:**
```
Data: [100, 105, 110, 115, 120, 125]

ARIMA(1,1,1):
- p=1: Use previous value
- d=1: Take first difference [5, 5, 5, 5, 5]
- q=1: Use previous error

Forecast: 130 (continuing the trend)
```

**Stationarity Test (ADF):**
- Stationary: Mean and variance don't change over time
- Non-stationary: Has trend or seasonality
- If p-value > 0.05: Non-stationary, need differencing (d=1)

**Why try multiple models?**
- Different data patterns need different parameters
- Use AIC (Akaike Information Criterion) to select best model
- Lower AIC = better model

**Alternatives:**
- Prophet: Facebook's forecasting library (handles seasonality better)
- SARIMA: Seasonal ARIMA (for monthly/quarterly data)
- Exponential Smoothing: Simpler, faster
- LSTM: Deep learning approach (needs more data)

**Potential Errors:**
- ❌ Insufficient data: Need at least 4 points
- ❌ Non-convergence: Model can't fit data
- ❌ Perfect correlation: Data is too regular

---

##### E. Fallback Regression (Lines 382-450)
```python
def simple_forecast_model(series: pd.Series, n_lags: int = 3) -> Dict[str, Any]:
    # Create lag features
    data = pd.DataFrame({"target": s})
    for i in range(1, n_lags + 1):
        data[f"lag_{i}"] = data["target"].shift(i)
    
    # Train linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
```

**What are lag features?**
Lag features use previous time steps as predictors.

**Example:**
```
Original series: [10, 15, 20, 25, 30]

With 3 lags:
  target | lag_1 | lag_2 | lag_3
  -------|-------|-------|-------
    20   |  15   |  10   |  NaN   (drop)
    25   |  20   |  15   |  10
    30   |  25   |  20   |  15

Model learns: target = f(lag_1, lag_2, lag_3)
Prediction: next = f(30, 25, 20) = 35
```

**Why use this as fallback?**
- Simpler than ARIMA
- More robust (fewer assumptions)
- Works with irregular data
- Faster to compute

**Alternatives:**
- Moving average: Simple average of last n values
- Exponential smoothing: Weighted average (recent values weighted more)
- Naive forecast: Next value = last value
- Seasonal naive: Next value = same period last year


---

##### F. Batch Forecasting Engine (Lines 452-550)
```python
def batch_forecast_backend(df, potential_time_cols, target_cols):
    for date_col in potential_time_cols:
        for target_col in target_cols:
            # Try ARIMA first
            arima_result = arima_forecast_students(series)
            if arima_result.get("status") == "ok":
                results[key] = arima_result
            else:
                # Fallback to regression
                simple_result = simple_forecast_model(series)
                results[key] = simple_result
```

**What is batch processing?**
- Automatically tries all combinations of time × target columns
- Example: 2 time columns × 5 numeric columns = 10 forecasts
- Returns metadata: successful, failed, skipped counts

**Why this approach?**
- User doesn't need to specify what to forecast
- System finds all possible predictions
- Graceful degradation (ARIMA → Regression → Skip)

**Metadata tracking:**
```python
metadata = {
    "total_combinations": 10,
    "successful_arima": 6,
    "fallback_regression": 3,
    "failed": 1,
    "skipped": 0
}
```

**Alternatives:**
- User-selected forecasts: Let user choose what to predict
- Single best forecast: Only return highest confidence prediction
- Parallel processing: Use multiprocessing for speed

---

##### G. RandomForest Training (Lines 552-620)
```python
def train_and_predict(df: pd.DataFrame, target_column: str):
    # Auto-detect task type
    is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 10
    
    if is_regression:
        model = RandomForestRegressor(n_estimators=50)
    else:
        model = RandomForestClassifier(n_estimators=50)
```

**What is RandomForest?**
- Ensemble of decision trees
- Each tree votes on prediction
- Majority vote wins (classification) or average (regression)

**Why RandomForest?**
- ✅ Handles mixed data types
- ✅ Resistant to overfitting
- ✅ No feature scaling needed
- ✅ Provides feature importance
- ❌ Slower than linear models
- ❌ Black box (hard to interpret)

**Regression vs Classification:**
```
Regression: Predict continuous values (price, temperature, count)
  Example: Predict student enrollment (1000, 1050, 1100...)

Classification: Predict categories (yes/no, high/medium/low)
  Example: Predict pass/fail (pass, fail)
```

**Auto-detection logic:**
- If numeric AND >10 unique values → Regression
- Otherwise → Classification

**Alternatives:**
- XGBoost: Usually more accurate, slower
- LightGBM: Faster, good for large datasets
- Neural Networks: Best for complex patterns, needs more data
- Linear Regression: Simpler, interpretable, less accurate

---

### 3️⃣ missing_value_handler.py - Data Cleaning

**Purpose**: Handle missing data using advanced imputation techniques.

#### Key Components:

##### A. Iterative Imputer (Lines 35-85)
```python
def iterative_impute(df, max_iter=10, random_state=42):
    imputer = IterativeImputer(
        max_iter=max_iter,
        random_state=random_state,
        n_nearest_features=n_nearest_features
    )
    imputed_numeric = imputer.fit_transform(numeric_data)
```

**What is Iterative Imputation?**
Also called MICE (Multiple Imputation by Chained Equations):

1. Start with simple imputation (mean/median)
2. For each column with missing values:
   - Use other columns to predict missing values
   - Update the column with predictions
3. Repeat until convergence (max_iter times)

**Example:**
```
Original:
  Age | Height | Weight
  25  |  170   |  NaN
  30  |  NaN   |  75
  NaN |  180   |  80

Iteration 1:
  Age | Height | Weight
  25  |  170   |  77.5  (mean)
  30  |  175   |  75    (mean)
  27.5|  180   |  80    (mean)

Iteration 2 (use regression):
  Age | Height | Weight
  25  |  170   |  70    (predicted from Age, Height)
  30  |  175   |  75    (predicted)
  28  |  180   |  80    (predicted)

... continues until stable
```

**Why better than simple imputation?**
- Considers relationships between columns
- More accurate than mean/median
- Preserves correlations in data

**Parameters:**
- `max_iter`: More iterations = more accurate but slower
- `n_nearest_features`: Use only closest features (faster)
- `random_state`: For reproducibility

**Alternatives:**
- Simple imputation: Mean, median, mode (fast but inaccurate)
- KNN imputation: Use k-nearest neighbors
- Forward/backward fill: Use previous/next value (time series)
- Drop rows: Remove rows with missing values (loses data)

**Potential Errors:**
- ❌ All values missing in column: Can't impute
- ❌ Too many missing values: Imputation unreliable
- ❌ Non-convergence: max_iter too low

---

##### B. Advanced Imputation with Encoding (Lines 87-160)
```python
def advanced_iterative_impute(df, max_iter=10):
    # Encode categorical variables
    for col in categorical_cols:
        le = LabelEncoder()
        df_work[col] = le.fit_transform(df[col])
    
    # Impute all columns
    imputed_data = imputer.fit_transform(df_work)
    
    # Decode back to original categories
    df_imputed[col] = le.inverse_transform(encoded_values)
```

**Why encode categorical variables?**
- Iterative Imputer only works with numbers
- Need to convert categories to numbers, then back

**Example:**
```
Original: ["Red", "Blue", NaN, "Green"]

Encode: [0, 1, NaN, 2]

Impute: [0, 1, 1, 2]  (predicted 1 for missing)

Decode: ["Red", "Blue", "Blue", "Green"]
```

**Label Encoding:**
- Red → 0
- Blue → 1
- Green → 2

**Alternatives:**
- One-hot encoding: Create binary columns (better for ML, but more columns)
- Target encoding: Use target variable statistics
- Frequency encoding: Use category frequency

---

### 4️⃣ visualization.py - Interactive Charts

**Purpose**: Create Power BI-style interactive visualizations.

#### Key Components:

##### A. PyGwalker Integration (Lines 18-35)
```python
def show_interactive_pygwalker(df):
    pyg_html = pyg.to_html(df, spec="./gw_config.json")
    components.html(pyg_html, height=1000, scrolling=True)
```

**What is PyGwalker?**
- Tableau-like interface in Python
- Drag-and-drop chart creation
- No coding required for users

**How it works:**
1. Converts DataFrame to interactive HTML
2. Embeds in Streamlit using components.html
3. User drags columns to create charts

**Alternatives:**
- Plotly Dash: More customizable, more code
- Tableau: Commercial, not embeddable
- Power BI: Commercial, not embeddable
- D3.js: Full control, requires JavaScript

---

##### B. Plotly Charts (Lines 37-800)

**Why Plotly?**
- Interactive by default (zoom, pan, hover)
- Beautiful out of the box
- Works in web browsers
- Supports animations

**Chart Types Implemented:**

1. **Bar Chart** (Lines 150-280)
   - Vertical/horizontal orientation
   - Color schemes
   - Value labels
   - Percentage calculations

2. **Pie Chart** (Lines 282-380)
   - Donut hole option
   - Pull out largest slice
   - Detailed breakdown table

3. **Histogram** (Lines 382-520)
   - Statistical overlays (mean, median, std)
   - Q-Q plot for normality
   - Box plot for outliers
   - Shapiro-Wilk test

4. **Scatter Plot** (Lines 522-680)
   - Trendline (OLS regression)
   - Color by category
   - Size by value
   - Marginal plots
   - Correlation metrics

5. **Correlation Heatmap** (Lines 682-800)
   - Hierarchical clustering
   - Interactive hover
   - Strength indicators
   - Variable analysis

**Concepts:**
- **Plotly Graph Objects**: Low-level API for full control
- **Plotly Express**: High-level API for quick charts
- **Subplots**: Multiple charts in one figure
- **Hover Templates**: Custom tooltip formatting

**Alternatives:**
- Matplotlib: Static charts, less interactive
- Seaborn: Beautiful statistical charts, less interactive
- Bokeh: Similar to Plotly, different API
- Altair: Declarative, Vega-based

---

## 🧠 Core Concepts Explained

### 1. Caching in Streamlit

**Problem**: Streamlit reruns entire script on every interaction

**Solution**: Cache expensive operations
```python
@st.cache_data  # For data (serializable)
def load_data():
    return pd.read_csv("large_file.csv")

@st.cache_resource  # For objects (non-serializable)
def load_model():
    return RandomForestClassifier()
```

**When to use:**
- File loading
- API calls
- Model training
- Data processing

**When NOT to use:**
- User input processing
- Random operations
- Time-dependent operations

---

### 2. RAG (Retrieval Augmented Generation)

**Traditional LLM:**
```
User: "What's the average enrollment?"
LLM: "I don't have access to your data"
```

**RAG System:**
```
User: "What's the average enrollment?"
System: 
  1. Search vector DB for relevant data
  2. Find: "2020: 1000, 2021: 1050, 2022: 1100"
  3. Send to LLM: "Given data [1000, 1050, 1100], answer: average enrollment?"
LLM: "The average enrollment is 1050 students"
```

**Components:**
1. **Embeddings**: Convert text to vectors
2. **Vector DB**: Store and search vectors
3. **Retriever**: Find relevant documents
4. **LLM**: Generate answer with context

---

### 3. Time Series Forecasting

**Components of Time Series:**
- **Trend**: Long-term increase/decrease
- **Seasonality**: Regular patterns (monthly, yearly)
- **Noise**: Random fluctuations

**Forecasting Methods:**

1. **ARIMA**: Best for stationary data with trend
2. **Seasonal ARIMA**: Handles seasonality
3. **Prophet**: Handles holidays and missing data
4. **LSTM**: Deep learning, needs lots of data

**Evaluation Metrics:**
- **MSE**: Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (more interpretable)
- **MAPE**: Mean Absolute Percentage Error (scale-independent)

---

### 4. Feature Engineering

**What is it?**
Creating new features from existing data to improve ML models.

**Examples:**
```python
# Year extraction
df['Year'] = pd.to_datetime(df['Date']).dt.year

# Binning
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 100])

# Lag features
df['Sales_Lag1'] = df['Sales'].shift(1)

# Interaction features
df['Price_Per_Unit'] = df['Total_Price'] / df['Quantity']
```

**Why important?**
- ML models can't create features
- Good features = better predictions
- Domain knowledge helps


---

## 🏗️ System Architecture

### Data Flow Diagram

```
┌─────────────────┐
│  User Uploads   │
│   CSV File      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         app.py (Main Controller)        │
│  ┌───────────────────────────────────┐  │
│  │  1. Load & Validate CSV           │  │
│  │  2. Transpose (optional)          │  │
│  │  3. Handle Missing Values         │  │
│  └───────────────┬───────────────────┘  │
└──────────────────┼──────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
┌──────────────────┐  ┌──────────────────┐
│  predictions.py  │  │ visualization.py │
│                  │  │                  │
│ • Detect columns │  │ • PyGwalker      │
│ • Cluster data   │  │ • Plotly charts  │
│ • ARIMA forecast │  │ • Statistics     │
│ • RandomForest   │  │                  │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌────────────────────┐
         │   LangChain RAG    │
         │                    │
         │ • CSVLoader        │
         │ • FAISS VectorDB   │
         │ • Gemini LLM       │
         └──────────┬─────────┘
                    │
                    ▼
         ┌────────────────────┐
         │  User Interface    │
         │                    │
         │ • Data preview     │
         │ • Visualizations   │
         │ • Predictions      │
         │ • Q&A chat         │
         └────────────────────┘
```

### Component Interaction

```python
# 1. User uploads CSV
uploaded_file = st.file_uploader("Upload CSV")

# 2. Save and load
df = pd.read_csv(uploaded_file)

# 3. Process with predictions
df_processed, mappings, forecasts, summary = process_csv_with_predictions(df)

# 4. Build RAG system
retriever = build_retriever(csv_path, api_key)

# 5. User asks question
answer = chain.run(user_question)

# 6. Display results
st.write(answer)
```

---

## ⚠️ Common Errors & Solutions

### 1. API Key Errors

**Error:**
```
❌ API key not found. Create secrets.json with your Gemini API key
```

**Solution:**
```json
// Create secrets.json in project root
{
  "GEMINI_API_KEY": "your_actual_api_key_here"
}
```

**How to get API key:**
1. Go to https://makersuite.google.com/app/apikey
2. Create new API key
3. Copy and paste into secrets.json

---

### 2. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
# Install all dependencies
pip install streamlit pandas numpy scikit-learn statsmodels
pip install langchain langchain-google-genai langchain-community
pip install plotly pygwalker faiss-cpu
```

**Or use requirements.txt:**
```bash
pip install -r requirements.txt
```

---

### 3. FAISS Memory Error

**Error:**
```
MemoryError: Unable to allocate array
```

**Cause:** CSV file too large for FAISS in-memory storage

**Solutions:**

**Option 1: Sample data**
```python
if len(df) > 10000:
    df_sample = df.sample(10000)
    retriever = build_retriever(df_sample)
```

**Option 2: Use Chroma (persistent storage)**
```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    docs, 
    embeddings,
    persist_directory="./chroma_db"
)
```

**Option 3: Increase system memory**
- Close other applications
- Use cloud instance with more RAM

---

### 4. ARIMA Convergence Error

**Error:**
```
ValueError: The computed initial AR coefficients are not stationary
```

**Cause:** Data is too irregular or has too few points

**Solution:** System automatically falls back to regression
```python
try:
    arima_result = arima_forecast_students(series)
except:
    simple_result = simple_forecast_model(series)  # Fallback
```

**Manual fix:**
```python
# Add more data points
# Remove outliers
# Use different ARIMA parameters
```

---

### 5. Streamlit Caching Issues

**Error:**
```
CachedObjectMutationWarning: Return value was mutated
```

**Cause:** Modifying cached object

**Solution:**
```python
# Wrong
@st.cache_data
def get_data():
    df = pd.read_csv("data.csv")
    return df

df = get_data()
df['new_col'] = 1  # ❌ Mutates cached object

# Right
@st.cache_data
def get_data():
    return pd.read_csv("data.csv")

df = get_data().copy()  # ✅ Create copy
df['new_col'] = 1
```

---

### 6. CSV Encoding Errors

**Error:**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solution:**
```python
# Try different encodings
df = pd.read_csv(file, encoding='utf-8')  # Default
df = pd.read_csv(file, encoding='latin-1')  # European
df = pd.read_csv(file, encoding='cp1252')  # Windows
df = pd.read_csv(file, encoding='iso-8859-1')  # Western European
```

---

## 🔄 Alternative Approaches

### 1. Instead of Streamlit

**Flask + React:**
```python
# Backend (Flask)
@app.route('/api/upload', methods=['POST'])
def upload_csv():
    file = request.files['file']
    df = pd.read_csv(file)
    return jsonify(process_data(df))

# Frontend (React)
function UploadCSV() {
  const handleUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();
    setResults(data);
  };
}
```

**Pros:**
- More control over UI
- Better for production
- Scalable

**Cons:**
- More code
- Need frontend skills
- Slower development

---

### 2. Instead of LangChain RAG

**Direct Pandas Filtering:**
```python
def answer_question(df, question):
    if "average" in question.lower():
        col = extract_column_name(question)
        return df[col].mean()
    elif "total" in question.lower():
        col = extract_column_name(question)
        return df[col].sum()
    # ... more rules
```

**Pros:**
- No API costs
- Faster
- More predictable

**Cons:**
- Limited to predefined questions
- No natural language understanding
- Lots of if-else code

---

### 3. Instead of ARIMA

**Prophet (Facebook):**
```python
from prophet import Prophet

# Prepare data
df_prophet = pd.DataFrame({
    'ds': dates,
    'y': values
})

# Fit model
model = Prophet()
model.fit(df_prophet)

# Forecast
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)
```

**Pros:**
- Handles seasonality automatically
- Works with missing data
- Intuitive parameters

**Cons:**
- Slower than ARIMA
- Less control
- Requires more data

---

### 4. Instead of RandomForest

**XGBoost:**
```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Pros:**
- Usually more accurate
- Faster training
- Better with large datasets

**Cons:**
- More hyperparameters to tune
- Can overfit easily
- Harder to interpret

---

### 5. Instead of Iterative Imputer

**KNN Imputer:**
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = imputer.fit_transform(df)
```

**Pros:**
- Simpler concept
- Faster
- Good for small datasets

**Cons:**
- Sensitive to scale
- Doesn't work well with categorical
- Needs feature scaling

---

## 📊 Visual Explanations

### How ARIMA Works

```
Step 1: Original Data (Non-stationary)
   │
   │     ╱
   │   ╱
   │ ╱
   └─────────────> Time

Step 2: Differencing (Make Stationary)
   │  ╱╲  ╱╲
   │╱    ╲╱  ╲
   └─────────────> Time

Step 3: Fit ARIMA Model
   - AR: Use past values
   - MA: Use past errors
   
Step 4: Forecast Future
   │     ╱
   │   ╱  ╱ (forecast)
   │ ╱  ╱
   └─────────────> Time
```

### How RAG Works

```
1. Document Processing
   CSV Row: "2023, Math, 1000 students"
      ↓
   Embedding: [0.23, -0.45, 0.67, ..., 0.12]
      ↓
   Vector DB: Store embedding

2. Query Processing
   User: "How many math students in 2023?"
      ↓
   Embedding: [0.25, -0.43, 0.65, ..., 0.10]
      ↓
   Similarity Search: Find closest vectors
      ↓
   Retrieved: "2023, Math, 1000 students"

3. Answer Generation
   LLM Input: 
     Context: "2023, Math, 1000 students"
     Question: "How many math students in 2023?"
      ↓
   LLM Output: "There were 1000 math students in 2023"
```

### How Clustering Works

```
Original Data (1000 unique ages):
18, 19, 20, 21, 22, 23, 24, 25, 26, 27, ...

After Binning (8 bins):
┌─────────┬─────────┬─────────┬─────────┐
│ 18-20   │ 21-23   │ 24-26   │ 27-29   │
│ (250)   │ (300)   │ (200)   │ (150)   │
└─────────┴─────────┴─────────┴─────────┘

Benefits:
✅ Reduced from 1000 to 8 categories
✅ Easier to visualize
✅ Better ML performance
✅ Handles outliers
```

---

## 🎓 Learning Path

### Beginner Level
1. **Python Basics**: Variables, functions, loops
2. **Pandas**: DataFrames, filtering, grouping
3. **Streamlit**: Basic widgets, layouts
4. **Plotly**: Simple charts

**Resources:**
- Python for Data Analysis (Book)
- Streamlit documentation
- Plotly tutorials

---

### Intermediate Level
1. **Machine Learning**: Scikit-learn basics
2. **Time Series**: ARIMA, forecasting
3. **Data Cleaning**: Missing values, outliers
4. **APIs**: REST APIs, authentication

**Resources:**
- Hands-On Machine Learning (Book)
- Kaggle courses
- Statsmodels documentation

---

### Advanced Level
1. **LLMs**: Transformers, embeddings
2. **RAG Systems**: Vector databases, retrieval
3. **Production**: Docker, deployment, monitoring
4. **Optimization**: Caching, async, profiling

**Resources:**
- LangChain documentation
- DeepLearning.AI courses
- Production ML systems

---

## 🚀 Deployment Guide

### Local Development
```bash
# 1. Clone repository
git clone <repo-url>
cd EduPredictv2

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add API key
echo '{"GEMINI_API_KEY": "your_key"}' > secrets.json

# 5. Run application
streamlit run app.py
```

---

### Streamlit Cloud (Free)
```bash
# 1. Push to GitHub
git init
git add .
git commit -m "Initial commit"
git push origin main

# 2. Go to share.streamlit.io
# 3. Connect GitHub repository
# 4. Add secrets in dashboard:
#    GEMINI_API_KEY = your_key
# 5. Deploy!
```

---

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

```bash
# Build and run
docker build -t edupredict .
docker run -p 8501:8501 edupredict
```

---

## 🔧 Customization Guide

### Add New Chart Type
```python
# In visualization.py

def show_custom_chart(df):
    st.subheader("My Custom Chart")
    
    # Your chart logic
    fig = px.line(df, x='date', y='value')
    st.plotly_chart(fig)

# In visualizations_sidebar()
plot_type = st.sidebar.radio(
    "Choose Visualization",
    [..., "🎨 Custom Chart"]
)

if plot_type == "🎨 Custom Chart":
    show_custom_chart(df)
```

---

### Add New Forecasting Method
```python
# In predictions.py

def prophet_forecast(series: pd.Series) -> Dict[str, Any]:
    from prophet import Prophet
    
    df = pd.DataFrame({
        'ds': series.index,
        'y': series.values
    })
    
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    
    return {
        "status": "ok",
        "method": "Prophet",
        "next_period_prediction": forecast['yhat'].iloc[-1]
    }

# In batch_forecast_backend()
# Add as another fallback option
```

---

### Add New Data Source
```python
# Support Excel files
uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])

if uploaded_file.name.endswith('.xlsx'):
    df = pd.read_excel(uploaded_file)
else:
    df = pd.read_csv(uploaded_file)

# Support databases
import sqlalchemy

engine = sqlalchemy.create_engine('postgresql://user:pass@host/db')
df = pd.read_sql("SELECT * FROM table", engine)
```

---

## 📈 Performance Optimization

### 1. Caching Strategy
```python
# Cache data loading
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(file):
    return pd.read_csv(file)

# Cache model training
@st.cache_resource
def train_model(df):
    model = RandomForestRegressor()
    model.fit(X, y)
    return model
```

### 2. Lazy Loading
```python
# Don't process everything upfront
if st.button("Run Predictions"):
    with st.spinner("Processing..."):
        results = process_predictions(df)
```

### 3. Sampling Large Datasets
```python
if len(df) > 100000:
    st.warning("Large dataset detected. Using sample for preview.")
    df_display = df.sample(10000)
else:
    df_display = df
```

---

## 🎯 Best Practices

### Code Organization
```python
# ✅ Good: Modular functions
def load_data(file):
    return pd.read_csv(file)

def clean_data(df):
    return df.dropna()

def analyze_data(df):
    return df.describe()

# ❌ Bad: Everything in one function
def do_everything(file):
    df = pd.read_csv(file)
    df = df.dropna()
    stats = df.describe()
    # ... 500 more lines
```

### Error Handling
```python
# ✅ Good: Specific error handling
try:
    df = pd.read_csv(file)
except FileNotFoundError:
    st.error("File not found")
except pd.errors.EmptyDataError:
    st.error("File is empty")
except Exception as e:
    st.error(f"Unexpected error: {e}")

# ❌ Bad: Generic catch-all
try:
    df = pd.read_csv(file)
except:
    st.error("Something went wrong")
```

### Documentation
```python
# ✅ Good: Clear docstring
def cluster_numeric_column(series: pd.Series, n_bins: int = 8) -> pd.Series:
    """
    Cluster numeric values into bins using quantile-based discretization.
    
    Args:
        series: Numeric pandas Series to cluster
        n_bins: Number of bins to create (default: 8)
    
    Returns:
        Series with bin labels as strings
    
    Example:
        >>> ages = pd.Series([18, 19, 20, 25, 30, 35])
        >>> cluster_numeric_column(ages, n_bins=3)
        0    (17.999, 20.667]
        1    (17.999, 20.667]
        2    (17.999, 20.667]
        3    (20.667, 27.333]
        4    (27.333, 35.0]
        5    (27.333, 35.0]
    """
    # Implementation
```

---

## 🎓 Conclusion

This project demonstrates:
- **Full-stack data application** development
- **Modern ML techniques** (ARIMA, RandomForest, RAG)
- **Production-ready code** (error handling, caching, modularity)
- **User-friendly interface** (Streamlit, interactive charts)

**Next Steps:**
1. Add more forecasting methods (Prophet, LSTM)
2. Implement user authentication
3. Add data export features
4. Create API endpoints
5. Add unit tests
6. Deploy to cloud

**Key Takeaways:**
- Start simple, add complexity gradually
- Cache expensive operations
- Handle errors gracefully
- Document your code
- Test with real data

---

## 📚 Additional Resources

### Documentation
- [Streamlit Docs](https://docs.streamlit.io)
- [Pandas Docs](https://pandas.pydata.org/docs)
- [Scikit-learn Docs](https://scikit-learn.org)
- [LangChain Docs](https://python.langchain.com)
- [Plotly Docs](https://plotly.com/python)

### Tutorials
- [Streamlit Tutorial](https://docs.streamlit.io/library/get-started)
- [Time Series Forecasting](https://otexts.com/fpp3)
- [RAG Systems](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data)

### Communities
- [Streamlit Forum](https://discuss.streamlit.io)
- [Kaggle](https://www.kaggle.com)
- [Stack Overflow](https://stackoverflow.com)

---

**Created by:** EduPredict Development Team  
**Last Updated:** 2024  
**Version:** 2.0  
**License:** MIT



---

## 📦 Libraries & Frameworks Used

### Core Framework
| Library | Version | Purpose | Why Used | Alternatives |
|---------|---------|---------|----------|--------------|
| **Streamlit** | Latest | Web UI framework | Rapid prototyping, no HTML/CSS needed | Flask, FastAPI, Dash, Gradio |

### Data Processing
| Library | Version | Purpose | Why Used | Alternatives |
|---------|---------|---------|----------|--------------|
| **Pandas** | Latest | Data manipulation | Industry standard, powerful DataFrame API | Polars, Dask, Vaex |
| **NumPy** | Latest | Numerical computing | Fast array operations, math functions | CuPy (GPU), JAX |

### Machine Learning
| Library | Version | Purpose | Why Used | Alternatives |
|---------|---------|---------|----------|--------------|
| **Scikit-learn** | Latest | ML algorithms | RandomForest, preprocessing, imputation | XGBoost, LightGBM, CatBoost |
| **Statsmodels** | Latest | Statistical models | ARIMA time series forecasting | Prophet, pmdarima, tsfresh |

### AI & LLM Integration
| Library | Version | Purpose | Why Used | Alternatives |
|---------|---------|---------|----------|--------------|
| **LangChain** | Latest | LLM framework | RAG implementation, chain management | LlamaIndex, Haystack, direct API |
| **LangChain-Google-GenAI** | Latest | Google Gemini integration | Access to Gemini models | OpenAI, Anthropic, Cohere |
| **LangChain-Community** | Latest | Community integrations | CSVLoader, FAISS support | Custom loaders |

### Vector Database
| Library | Version | Purpose | Why Used | Alternatives |
|---------|---------|---------|----------|--------------|
| **FAISS** | CPU version | Vector similarity search | Fast, in-memory, free | Pinecone, Weaviate, Chroma, Milvus |

### Visualization
| Library | Version | Purpose | Why Used | Alternatives |
|---------|---------|---------|----------|--------------|
| **Plotly** | Latest | Interactive charts | Beautiful, interactive, web-based | Matplotlib, Seaborn, Bokeh, Altair |
| **PyGwalker** | Latest | Tableau-like interface | Drag-and-drop visualization | Tableau, Power BI, Looker |
| **Matplotlib** | Latest | Static plots | Backend for other libraries | - |
| **Seaborn** | Latest | Statistical visualization | Beautiful statistical plots | - |

### Utilities
| Library | Version | Purpose | Why Used | Alternatives |
|---------|---------|---------|----------|--------------|
| **asyncio** | Built-in | Async operations | Required by Streamlit + LangChain | threading, multiprocessing |
| **pathlib** | Built-in | Path handling | Cross-platform path operations | os.path |
| **json** | Built-in | JSON parsing | Configuration file handling | yaml, toml |
| **re** | Built-in | Regular expressions | Pattern matching (year extraction) | - |
| **warnings** | Built-in | Warning suppression | Clean output | - |

---

## 📚 Detailed Library Explanations

### 1. Streamlit
```python
import streamlit as st
```

**What it does:**
- Creates web apps with pure Python
- Auto-reloads on code changes
- Built-in widgets (buttons, sliders, file uploaders)

**Key Features Used:**
- `st.file_uploader()`: Upload CSV files
- `st.dataframe()`: Display interactive tables
- `st.plotly_chart()`: Embed Plotly charts
- `st.cache_data()`: Cache data operations
- `st.cache_resource()`: Cache model objects
- `st.sidebar`: Create sidebar widgets
- `st.columns()`: Multi-column layouts
- `st.spinner()`: Loading indicators

**Installation:**
```bash
pip install streamlit
```

**Alternatives:**
- **Flask**: More control, requires HTML/CSS
- **FastAPI**: API-first, modern async
- **Dash**: Plotly's framework, more verbose
- **Gradio**: Similar to Streamlit, ML-focused

---

### 2. Pandas
```python
import pandas as pd
```

**What it does:**
- DataFrame operations (filter, group, merge)
- Data cleaning and transformation
- CSV/Excel reading and writing

**Key Features Used:**
- `pd.read_csv()`: Load CSV files
- `pd.DataFrame()`: Create data structures
- `df.groupby()`: Aggregate data
- `df.describe()`: Statistical summary
- `df.isnull()`: Detect missing values
- `pd.qcut()`: Quantile-based binning
- `pd.cut()`: Range-based binning

**Installation:**
```bash
pip install pandas
```

**Alternatives:**
- **Polars**: Faster, Rust-based
- **Dask**: Parallel computing, larger-than-memory
- **Vaex**: Out-of-core DataFrames

---

### 3. Scikit-learn
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
```

**What it does:**
- Machine learning algorithms
- Data preprocessing
- Model evaluation

**Key Features Used:**
- **RandomForest**: Ensemble learning (classification/regression)
- **IterativeImputer**: MICE algorithm for missing values
- **LabelEncoder**: Convert categories to numbers
- **train_test_split**: Split data for validation
- **Metrics**: Evaluate model performance

**Installation:**
```bash
pip install scikit-learn
```

**Alternatives:**
- **XGBoost**: Gradient boosting, often more accurate
- **LightGBM**: Fast gradient boosting
- **CatBoost**: Handles categorical data natively

---

### 4. Statsmodels
```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
```

**What it does:**
- Statistical modeling
- Time series analysis
- Hypothesis testing

**Key Features Used:**
- **ARIMA**: AutoRegressive Integrated Moving Average
- **adfuller**: Augmented Dickey-Fuller test (stationarity)

**Installation:**
```bash
pip install statsmodels
```

**Alternatives:**
- **Prophet**: Facebook's forecasting library
- **pmdarima**: Auto ARIMA parameter selection
- **tsfresh**: Automated feature extraction

---

### 5. LangChain
```python
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
```

**What it does:**
- Framework for LLM applications
- RAG (Retrieval Augmented Generation)
- Chain multiple LLM calls

**Key Features Used:**
- **ChatGoogleGenerativeAI**: Gemini LLM interface
- **GoogleGenerativeAIEmbeddings**: Text to vectors
- **CSVLoader**: Load CSV as documents
- **FAISS**: Vector database integration
- **RetrievalQA**: Question-answering chain

**Installation:**
```bash
pip install langchain langchain-google-genai langchain-community
```

**Alternatives:**
- **LlamaIndex**: Similar RAG framework
- **Haystack**: NLP framework by deepset
- **Direct API calls**: More control, more code

---

### 6. FAISS
```python
from langchain_community.vectorstores import FAISS
```

**What it does:**
- Fast similarity search
- Vector database (in-memory)
- Efficient nearest neighbor search

**Key Features Used:**
- `FAISS.from_documents()`: Create vector store
- `as_retriever()`: Search interface

**Installation:**
```bash
pip install faiss-cpu  # CPU version
# OR
pip install faiss-gpu  # GPU version (faster)
```

**Alternatives:**
- **Chroma**: Persistent storage, easier API
- **Pinecone**: Cloud-based, scalable
- **Weaviate**: GraphQL API, production-ready
- **Milvus**: Open-source, distributed

---

### 7. Plotly
```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

**What it does:**
- Interactive web-based charts
- Hover tooltips, zoom, pan
- Export to PNG/SVG

**Key Features Used:**
- **plotly.express**: High-level API (quick charts)
- **plotly.graph_objects**: Low-level API (full control)
- **make_subplots**: Multiple charts in one figure

**Chart Types Used:**
- Bar charts (vertical/horizontal)
- Pie charts (with donut hole)
- Histograms (with statistical overlays)
- Scatter plots (with trendlines)
- Heatmaps (correlation matrices)
- Box plots (outlier detection)

**Installation:**
```bash
pip install plotly
```

**Alternatives:**
- **Matplotlib**: Static charts, more control
- **Seaborn**: Statistical plots, beautiful defaults
- **Bokeh**: Similar to Plotly, different API
- **Altair**: Declarative, Vega-based

---

### 8. PyGwalker
```python
import pygwalker as pyg
```

**What it does:**
- Tableau-like interface in Python
- Drag-and-drop visualization
- No coding required for users

**Key Features Used:**
- `pyg.to_html()`: Generate interactive HTML
- Drag columns to axes
- Switch chart types dynamically

**Installation:**
```bash
pip install pygwalker
```

**Alternatives:**
- **Tableau**: Commercial, not embeddable
- **Power BI**: Commercial, not embeddable
- **Looker**: Commercial, cloud-based
- **Metabase**: Open-source BI tool

---

## 🔧 Complete Installation Guide

### Method 1: pip (Standard)
```bash
# Core dependencies
pip install streamlit pandas numpy

# Machine learning
pip install scikit-learn statsmodels

# AI & LLM
pip install langchain langchain-google-genai langchain-community

# Vector database
pip install faiss-cpu

# Visualization
pip install plotly pygwalker matplotlib seaborn

# Utilities
pip install python-dotenv  # Optional: for .env files
```

### Method 2: requirements.txt
```txt
# requirements.txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
langchain>=0.1.0
langchain-google-genai>=0.0.6
langchain-community>=0.0.13
faiss-cpu>=1.7.4
plotly>=5.17.0
pygwalker>=0.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

```bash
pip install -r requirements.txt
```

### Method 3: Poetry (Modern)
```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.10"
streamlit = "^1.28.0"
pandas = "^2.0.0"
numpy = "^1.24.0"
scikit-learn = "^1.3.0"
statsmodels = "^0.14.0"
langchain = "^0.1.0"
langchain-google-genai = "^0.0.6"
langchain-community = "^0.0.13"
faiss-cpu = "^1.7.4"
plotly = "^5.17.0"
pygwalker = "^0.3.0"
matplotlib = "^3.7.0"
seaborn = "^0.12.0"
```

```bash
poetry install
```

### Method 4: Conda
```bash
conda create -n edupredict python=3.10
conda activate edupredict
conda install -c conda-forge streamlit pandas numpy scikit-learn statsmodels plotly matplotlib seaborn
pip install langchain langchain-google-genai langchain-community faiss-cpu pygwalker
```

---

## 🎯 Library Usage Statistics

### Import Frequency in Project
```
streamlit:           50+ uses (UI, widgets, caching)
pandas:              100+ uses (data manipulation)
numpy:               20+ uses (numerical operations)
scikit-learn:        30+ uses (ML models, preprocessing)
statsmodels:         10+ uses (ARIMA forecasting)
langchain:           15+ uses (RAG system)
plotly:              40+ uses (visualizations)
pygwalker:           2 uses (interactive dashboard)
```

### Performance Impact
```
Fastest:    pandas, numpy (optimized C/Cython)
Medium:     scikit-learn, plotly
Slow:       ARIMA (iterative fitting)
Slowest:    LLM API calls (network latency)
```

### Memory Usage
```
Low:        streamlit, numpy
Medium:     pandas, scikit-learn
High:       FAISS (stores all embeddings)
Variable:   plotly (depends on data size)
```

---

## 🔄 Library Upgrade Guide

### Check Current Versions
```bash
pip list | grep streamlit
pip list | grep pandas
pip list | grep scikit-learn
```

### Upgrade All
```bash
pip install --upgrade streamlit pandas numpy scikit-learn statsmodels
pip install --upgrade langchain langchain-google-genai langchain-community
pip install --upgrade plotly pygwalker matplotlib seaborn
```

### Upgrade Specific Library
```bash
pip install --upgrade streamlit
```

### Check for Security Issues
```bash
pip install safety
safety check
```

---

## 🐛 Common Library Issues

### Issue 1: FAISS Installation Fails
```bash
# Error: No matching distribution found for faiss
# Solution: Use faiss-cpu or faiss-gpu
pip install faiss-cpu
```

### Issue 2: LangChain Import Errors
```bash
# Error: cannot import name 'ChatGoogleGenerativeAI'
# Solution: Install correct package
pip install langchain-google-genai
```

### Issue 3: Plotly Not Displaying
```bash
# Error: Charts not showing in Streamlit
# Solution: Use st.plotly_chart() not fig.show()
st.plotly_chart(fig, use_container_width=True)
```

### Issue 4: PyGwalker HTML Not Rendering
```bash
# Error: PyGwalker interface blank
# Solution: Use components.html with scrolling
import streamlit.components.v1 as components
components.html(pyg_html, height=1000, scrolling=True)
```

---

## 📊 Library Comparison Matrix

| Feature | Streamlit | Dash | Gradio | Flask |
|---------|-----------|------|--------|-------|
| Learning Curve | Easy | Medium | Easy | Hard |
| Customization | Medium | High | Low | Very High |
| Speed | Fast | Medium | Fast | Fast |
| Deployment | Easy | Medium | Easy | Medium |
| Best For | Data Apps | Dashboards | ML Demos | APIs |

| Feature | Pandas | Polars | Dask | Spark |
|---------|--------|--------|------|-------|
| Speed | Fast | Faster | Fast | Very Fast |
| Memory | Medium | Low | Low | Distributed |
| API | Mature | New | Similar | Different |
| Best For | General | Performance | Big Data | Huge Data |

| Feature | Scikit-learn | XGBoost | LightGBM | TensorFlow |
|---------|--------------|---------|----------|------------|
| Ease of Use | Easy | Medium | Medium | Hard |
| Speed | Medium | Fast | Very Fast | Fast |
| Accuracy | Good | Better | Better | Best |
| Best For | General ML | Competitions | Production | Deep Learning |

---

## 🎓 Learning Resources for Each Library

### Streamlit
- Official Docs: https://docs.streamlit.io
- Tutorial: https://docs.streamlit.io/library/get-started
- Gallery: https://streamlit.io/gallery

### Pandas
- Official Docs: https://pandas.pydata.org/docs
- 10 Minutes to Pandas: https://pandas.pydata.org/docs/user_guide/10min.html
- Cheat Sheet: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

### Scikit-learn
- Official Docs: https://scikit-learn.org/stable
- Tutorials: https://scikit-learn.org/stable/tutorial/index.html
- Examples: https://scikit-learn.org/stable/auto_examples/index.html

### LangChain
- Official Docs: https://python.langchain.com
- Cookbook: https://github.com/langchain-ai/langchain/tree/master/cookbook
- DeepLearning.AI Course: https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data

### Plotly
- Official Docs: https://plotly.com/python
- Gallery: https://plotly.com/python/plotly-express
- Cheat Sheet: https://images.plot.ly/plotly-documentation/images/python_cheat_sheet.pdf

---

**Summary:** This project uses **15+ libraries** spanning web frameworks, data processing, machine learning, AI/LLM integration, vector databases, and visualization - creating a complete, production-ready data analysis platform.

