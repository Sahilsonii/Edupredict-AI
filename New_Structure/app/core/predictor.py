# predictor.py - Complete ML Predictions with Clustering & Batch Processing

import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import math
import warnings
from ..utils.helpers import EDUCATIONAL_KEYWORDS, ALL_EDUCATIONAL_KEYWORDS, YEAR_REGEX, extract_year_from_string

warnings.filterwarnings('ignore')

def round_years_in_column(series: pd.Series) -> pd.Series:
    """Convert Series to integer years. Examples: '2000/01' -> 2000"""
    extracted = series.astype(str).apply(lambda x: extract_year_from_string(x))
    return pd.Series(extracted, index=series.index, dtype="Int64")

def cluster_numeric_column(series: pd.Series, n_bins: int = 8) -> pd.Series:
    """Cluster numeric values into bins."""
    s = series.dropna()
    if s.empty or len(s) < 2:
        return pd.Series(["unknown"] * len(series), index=series.index, dtype="object")

    try:
        if s.nunique() > n_bins:
            bins = pd.qcut(s, q=n_bins, duplicates="drop")
        else:
            bins = pd.cut(s, bins=min(n_bins, s.nunique()))

        labels = pd.Series("unknown", index=series.index, dtype="object")
        labels.loc[bins.index] = bins.astype(str)
        return labels
    except Exception:
        return pd.Series(["unknown"] * len(series), index=series.index, dtype="object")

def cluster_categorical_column(series: pd.Series, top_k: int = 15) -> pd.Series:
    """Keep top_k categories, group others as 'Other'."""
    s = series.fillna("MISSING").astype(str)
    top_categories = s.value_counts().nlargest(top_k).index.tolist()
    clustered = s.apply(lambda x: x if x in top_categories else "Other")
    return clustered

def cluster_all_columns(
    df: pd.DataFrame,
    numeric_bins: int = 8,
    cat_top_k: int = 15,
    treat_years: bool = True,
    keep_original: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Apply intelligent clustering to ALL columns."""
    df_proc = df.copy()
    mappings = {}

    for col in df.columns:
        col_series = df[col]

        # 1) Year Detection & Clustering
        if treat_years:
            name_hints = any(hint in col.lower() for hint in ["year", "date", "period", "time"])
            sample_vals = col_series.dropna().astype(str).head(30).tolist()
            value_hints = sum(1 for s in sample_vals if YEAR_REGEX.search(s))

            if name_hints or value_hints >= len(sample_vals) // 4:
                year_series = round_years_in_column(col_series)
                if year_series.notna().sum() >= max(3, len(col_series) // 10):
                    cluster_col = f"{col}__year" if keep_original else col
                    df_proc[cluster_col] = year_series

                    unique_years = sorted([int(x) for x in year_series.dropna().unique()])
                    mappings[col] = {
                        "type": "year",
                        "cluster_col": cluster_col,
                        "unique_clusters": unique_years,
                        "description": f"Year clustering: {len(unique_years)} unique years"
                    }
                    continue

        # 2) Numeric Clustering
        if pd.api.types.is_numeric_dtype(col_series):
            cluster_col = f"{col}__bin" if keep_original else col
            labels = cluster_numeric_column(col_series, n_bins=numeric_bins)
            df_proc[cluster_col] = labels

            unique_bins = labels.dropna().unique().tolist()
            mappings[col] = {
                "type": "numeric",
                "cluster_col": cluster_col,
                "unique_clusters": unique_bins,
                "description": f"Numeric binning: {len(unique_bins)} bins"
            }
            continue

        # 3) Categorical Clustering
        cluster_col = f"{col}__cat" if keep_original else col
        labels = cluster_categorical_column(col_series, top_k=cat_top_k)
        df_proc[cluster_col] = labels

        unique_cats = labels.unique().tolist()
        mappings[col] = {
            "type": "categorical", 
            "cluster_col": cluster_col,
            "unique_clusters": unique_cats,
            "description": f"Categorical grouping: {len(unique_cats)} groups"
        }

    return df_proc, mappings

def simple_forecast_model(series: pd.Series, n_lags: int = 3) -> Dict[str, Any]:
    """Simple lag-based forecasting."""
    try:
        s = series.dropna().sort_index()
        if len(s) < 6:
            return {"status": "insufficient_data"}

        # Create lag features
        data = pd.DataFrame({"target": s})
        for i in range(1, n_lags + 1):
            data[f"lag_{i}"] = data["target"].shift(i)

        data = data.dropna()
        if len(data) < 3:
            return {"status": "insufficient_data"}

        X = data[[f"lag_{i}" for i in range(1, n_lags + 1)]]
        y = data["target"]

        split_idx = max(1, len(data) - 2)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = LinearRegression()
        model.fit(X_train, y_train)

        if len(X_test) > 0:
            test_preds = model.predict(X_test)
            mse = mean_squared_error(y_test, test_preds)
        else:
            test_preds = []
            mse = 0.0

        # Forecast next period
        last_values = X.iloc[-1].values.reshape(1, -1)
        next_prediction = float(model.predict(last_values)[0])

        return {
            "status": "ok",
            "method": "LAG_REGRESSION",
            "mse": mse,
            "next_period_prediction": next_prediction,
            "periods": data.index.tolist(),
            "y_values": y.tolist(),
            "predictions_test": test_preds.tolist(),
            "actuals_test": y_test.tolist()
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

def aggregate_by_period_for_target(df: pd.DataFrame, date_col: str, target_col: str) -> pd.Series:
    """Aggregate target by year periods."""
    years = round_years_in_column(df[date_col])
    temp_df = pd.DataFrame({
        'year': years,
        'target': pd.to_numeric(df[target_col], errors='coerce')
    }).dropna()

    if len(temp_df) < 4:
        return pd.Series([], dtype=float)

    yearly_data = temp_df.groupby('year')['target'].sum().sort_index()
    return yearly_data

def round_to_historical_pattern(prediction: float, historical_data: pd.Series) -> int:
    """Round predictions based on historical patterns"""
    # Get common rounding patterns from historical data
    differences = historical_data.diff().dropna()
    median_change = abs(differences.median())
    
    # Find most common last digits
    last_digits = [int(str(abs(int(x)))[-2:]) for x in historical_data if not pd.isna(x)]
    if last_digits:
        common_last_digits = pd.Series(last_digits).mode()[0]
    else:
        common_last_digits = 0
    
    # Round to nearest multiple of pattern
    if median_change > 100:
        base = 10 ** (len(str(int(median_change))) - 2)
        rounded = round(prediction / base) * base
    else:
        rounded = round(prediction)
    
    # Adjust last two digits to match historical pattern
    rounded_str = str(int(rounded))
    if len(rounded_str) >= 2:
        rounded_str = rounded_str[:-2] + str(common_last_digits).zfill(2)
        rounded = int(rounded_str)
    
    return rounded

def arima_forecast_students(series: pd.Series) -> Dict[str, Any]:
    """✅ Robust ARIMA forecasting with automatic model selection"""
    try:
        s = series.dropna().sort_index()
        if len(s) < 4:
            return {"status": "insufficient_data", "detail": "Need at least 4 data points"}

        # Test for stationarity
        try:
            adf_test = adfuller(s)
            d = 1 if adf_test[1] > 0.05 else 0
        except:
            d = 1

        # Try different ARIMA models
        best_aic = float('inf')
        best_model = None
        best_order = None

        for p in range(3):
            for q in range(3):
                try:
                    model = ARIMA(s, order=(p, d, q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_model = results
                        best_order = (p, d, q)
                except:
                    continue

        if best_model is None:
            return {"status": "model_error", "detail": "ARIMA model fitting failed"}

        # Make forecast
        forecast = best_model.forecast(steps=1)
        next_period_pred = forecast[0]

        # Round prediction based on historical patterns
        rounded_prediction = round_to_historical_pattern(next_period_pred, s)

        # Get confidence intervals
        conf_int = best_model.get_forecast(steps=1).conf_int()
        lower_bound = round_to_historical_pattern(conf_int.iloc[0, 0], s)
        upper_bound = round_to_historical_pattern(conf_int.iloc[0, 1], s)

        return {
            "status": "ok",
            "method": f"ARIMA{best_order}",
            "next_period_prediction": rounded_prediction,
            "confidence_interval": {
                "lower": lower_bound,
                "upper": upper_bound
            },
            "historical_data": {
                "periods": s.index.tolist(),
                "values": s.tolist()
            },
            "model_info": {
                "aic": best_model.aic,
                "order": best_order
            }
        }

    except Exception as e:
        return {"status": "error", "error": str(e), "detail": "ARIMA forecasting failed"}

def detect_educational_context(df: pd.DataFrame) -> Dict[str, Any]:
    """✅ Detect if dataset is educational and identify relevant categories"""
    col_names = ' '.join(df.columns).lower()
    
    # Check column names for educational keywords
    detected_categories = {}
    total_matches = 0
    
    for category, keywords in EDUCATIONAL_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in col_names]
        if matches:
            detected_categories[category] = matches
            total_matches += len(matches)
    
    # Check data values for educational keywords (sample)
    value_matches = 0
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_values = ' '.join(df[col].dropna().astype(str).head(50).tolist()).lower()
            if any(kw in sample_values for kw in ALL_EDUCATIONAL_KEYWORDS):
                value_matches += 1
    
    is_educational = total_matches > 0 or value_matches > 0
    
    return {
        'is_educational': is_educational,
        'confidence': min(100, (total_matches * 10) + (value_matches * 5)),
        'detected_categories': detected_categories,
        'total_keyword_matches': total_matches,
        'columns_with_edu_values': value_matches
    }

def auto_detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """✅ Auto-detect year, numeric, and categorical columns with educational context"""
    year_cols = []
    numeric_cols = []
    categorical_cols = []
    educational_cols = []
    
    for col in df.columns:
        col_series = df[col]
        col_lower = col.lower()
        
        # Check if column is educational
        if any(kw in col_lower for kw in ALL_EDUCATIONAL_KEYWORDS):
            educational_cols.append(col)
        
        # Check for year columns
        name_hints = any(hint in col_lower for hint in ["year", "date", "period", "time", "semester", "term", "session"])
        sample_vals = col_series.dropna().astype(str).head(30).tolist()
        value_hints = sum(1 for s in sample_vals if YEAR_REGEX.search(s))
        
        if name_hints or value_hints >= len(sample_vals) // 4:
            year_series = round_years_in_column(col_series)
            if year_series.notna().sum() >= max(3, len(col_series) // 10):
                year_cols.append(col)
                continue
        
        # Check for numeric columns
        if pd.api.types.is_numeric_dtype(col_series):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return {
        "year": year_cols,
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "educational": educational_cols
    }

def detect_numerical_columns(df: pd.DataFrame) -> List[str]:
    """Detect all numerical columns in the dataset"""
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

def detect_time_columns(df: pd.DataFrame) -> List[str]:
    """Detect time-related columns in the dataset"""
    time_keywords = ['year', 'date', 'period', 'time', 'month', 'day']
    time_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        # Check if column name contains time-related keywords
        if any(keyword in col_lower for keyword in time_keywords):
            time_cols.append(col)
            continue
        
        # Check if column values look like years
        sample_values = df[col].dropna().astype(str).head(10).tolist()
        year_count = sum(1 for val in sample_values if YEAR_REGEX.search(str(val)))
        if year_count >= len(sample_values) // 2:
            time_cols.append(col)
    
    return time_cols

def batch_forecast_backend(
    df: pd.DataFrame, 
    potential_time_cols: Optional[List[str]] = None,
    target_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """✅ Generate forecasts with ARIMA + fallback regression"""
    
    results = {}
    metadata = {
        "total_combinations": 0,
        "successful_arima": 0,
        "fallback_regression": 0,
        "failed": 0,
        "skipped": 0
    }
    
    try:
        # Auto-detect time columns if not provided
        if potential_time_cols is None:
            potential_time_cols = detect_time_columns(df)
        
        if not potential_time_cols:
            return {
                "status": "error", 
                "error": "No time-related columns found",
                "detail": "Dataset must contain year/date columns"
            }
        
        # Auto-detect target columns if not provided
        if target_cols is None:
            target_cols = [
                col for col in df.columns 
                if col not in potential_time_cols and pd.api.types.is_numeric_dtype(df[col])
            ]
        
        if not target_cols:
            return {
                "status": "error", 
                "error": "No numeric target columns found",
                "detail": "Dataset must contain numeric columns for forecasting"
            }
        
        # Generate forecasts for each combination
        for date_col in potential_time_cols:
            for target_col in target_cols:
                if date_col == target_col:
                    continue

                metadata["total_combinations"] += 1
                key = f"{target_col}__by__{date_col}"
                
                try:
                    # Get time series data
                    series = aggregate_by_period_for_target(df, date_col, target_col)
                    
                    if len(series) < 4:
                        results[key] = {
                            "status": "skipped",
                            "reason": "Insufficient data points for forecasting",
                            "data_points": len(series),
                            "detail": "Need at least 4 time periods"
                        }
                        metadata["skipped"] += 1
                        continue

                    # Try ARIMA first, fallback to simple model
                    arima_result = arima_forecast_students(series)
                    if arima_result.get("status") == "ok":
                        results[key] = arima_result
                        metadata["successful_arima"] += 1
                    else:
                        # Fallback to regression
                        simple_result = simple_forecast_model(series)
                        if simple_result.get("status") == "ok":
                            simple_result["fallback"] = True
                            simple_result["arima_error"] = arima_result.get("detail", "ARIMA failed")
                            results[key] = simple_result
                            metadata["fallback_regression"] += 1
                        else:
                            results[key] = simple_result
                            metadata["failed"] += 1

                except Exception as e:
                    results[key] = {
                        "status": "error",
                        "error": str(e),
                        "target": target_col,
                        "date_col": date_col,
                        "detail": "Forecasting failed for this combination"
                    }
                    metadata["failed"] += 1
        
        results["_metadata"] = metadata
        return results
    
    except Exception as e:
        return {
            "status": "error", 
            "error": f"Batch forecasting failed: {str(e)}",
            "detail": "Critical error in batch processing"
        }

def get_column_forecast(batch_results: Dict[str, Any], target_col: str, date_col: Optional[str] = None) -> Dict[str, Any]:
    """Get forecast results for specific column."""
    if date_col:
        key = f"{target_col}__by__{date_col}"
        return batch_results.get(key, {"status": "not_found"})
    else:
        # Return all forecasts for this target column
        matches = {k: v for k, v in batch_results.items() 
                  if k.startswith(f"{target_col}__by__")}
        return matches if matches else {"status": "not_found"}

def get_all_successful_predictions(batch_results: Dict[str, Any]) -> Dict[str, List[str]]:
    """Get summary of successful predictions."""
    forecasts = []
    
    for key, result in batch_results.items():
        if result.get('status') == 'ok':
            forecasts.append(key)
    
    return {
        "ml_predictions": [],  # Not implemented in this version
        "time_forecasts": forecasts,
        "total_successful": len(forecasts)
    }

# Basic ML prediction functions (for compatibility)
def preprocess_data(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Basic preprocessing."""
    y = df[target_column]
    X = df.drop(columns=[target_column])
    X = X.dropna()
    y = y.loc[X.index]
    X = pd.get_dummies(X, drop_first=True)
    
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype('category').cat.codes
    
    return X, y

def train_and_predict(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """✅ Train RandomForest for regression/classification with detailed metrics"""
    try:
        X, y = preprocess_data(df, target_column)
        
        if len(X) < 10:
            raise ValueError("Not enough data after preprocessing (need at least 10 samples)")
        
        # Auto-detect task type
        is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 10
        
        if is_regression:
            model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
            metric_name = "Mean Squared Error"
            task_type = "regression"
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
            metric_name = "Accuracy"
            task_type = "classification"
        
        test_size = min(0.3, max(0.1, 0.2))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        if isinstance(model, RandomForestRegressor):
            metric_value = mean_squared_error(y_test, predictions)
        else:
            metric_value = accuracy_score(y_test, predictions)
        
        return {
            'status': 'ok',
            'model': model,
            'task_type': task_type,
            'metric_name': metric_name,
            'metric_value': metric_value,
            'predictions': predictions,
            'actuals': y_test.values,
            'feature_names': X.columns.tolist(),
            'n_samples': len(X),
            'n_features': len(X.columns),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'error': f"Prediction failed: {str(e)}",
            'detail': 'RandomForest training failed'
        }

def fit_arima_model(series: pd.Series) -> Dict[str, Any]:
    """Fit ARIMA model and generate forecast"""
    if len(series) < 4:
        return {"status": "error", "error": "Insufficient data points"}
    
    try:
        # Fit ARIMA(1,1,1) model
        model = ARIMA(series, order=(1,1,1))
        results = model.fit()
        
        # Generate forecast
        forecast = results.forecast(steps=1)
        conf_int = results.get_forecast(steps=1).conf_int()
        
        return {
            "status": "ok",
            "method": "ARIMA(1,1,1)",
            "next_period_prediction": float(round(forecast[0], 2)),
            "confidence_interval": {
                "lower": float(round(conf_int.iloc[0, 0], 2)),
                "upper": float(round(conf_int.iloc[0, 1], 2))
            },
            "mse": ((series - results.fittedvalues) ** 2).mean()
        }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}
