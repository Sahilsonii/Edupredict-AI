# missing_value_handler.py

import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# --- Display summary of missing values ---
def show_missing_summary(df):
    """
    Display a table showing missing count and percentage per column
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df) * 100).round(2)
    summary = pd.DataFrame({"Missing Count": missing_count, "Missing %": missing_percent})
    summary = summary[summary["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
    
    if len(summary) > 0:
        st.subheader("🔍 Missing Values Summary")
        st.dataframe(summary, use_container_width=True)
        return summary
    else:
        st.success("✅ No missing values found in the dataset!")
        return pd.DataFrame()

# --- Iterative Imputer for missing values ---
def iterative_impute(df, max_iter=10, random_state=42, n_nearest_features=None):
    """
    Use Iterative Imputer to fill missing values using machine learning approach
    
    Parameters:
    - df: DataFrame with missing values
    - max_iter: Maximum number of imputation rounds
    - random_state: Random state for reproducibility
    - n_nearest_features: Number of features to use for imputation
    
    Returns:
    - DataFrame with imputed values
    """
    if df.isnull().sum().sum() == 0:
        st.info("✅ No missing values to impute!")
        return df
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    df_imputed = df.copy()
    
    # Handle numeric columns with Iterative Imputer
    if numeric_cols:
        st.write("🔢 **Imputing numeric columns:**", numeric_cols)
        
        # Configure Iterative Imputer
        imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=random_state,
            n_nearest_features=n_nearest_features
        )
        
        # Fit and transform numeric data
        numeric_data = df[numeric_cols]
        imputed_numeric = imputer.fit_transform(numeric_data)
        
        # Replace in dataframe
        df_imputed[numeric_cols] = imputed_numeric
        
        st.success(f"✅ Imputed {len(numeric_cols)} numeric columns")
    
    # Handle categorical columns with mode imputation
    if categorical_cols:
        st.write("📝 **Imputing categorical columns:**", categorical_cols)
        
        for col in categorical_cols:
            if df_imputed[col].isnull().sum() > 0:
                mode_value = df_imputed[col].mode()
                if len(mode_value) > 0:
                    df_imputed[col].fillna(mode_value[0], inplace=True)
                else:
                    df_imputed[col].fillna("Unknown", inplace=True)
        
        st.success(f"✅ Imputed {len(categorical_cols)} categorical columns with mode")
    
    return df_imputed

# --- Advanced Iterative Imputer with encoding ---
def advanced_iterative_impute(df, max_iter=10, random_state=42):
    """
    Advanced iterative imputation that handles both numeric and categorical data
    by encoding categorical variables first
    
    Parameters:
    - df: DataFrame with missing values
    - max_iter: Maximum number of imputation rounds
    - random_state: Random state for reproducibility
    
    Returns:
    - DataFrame with imputed values and original data types preserved
    """
    if df.isnull().sum().sum() == 0:
        st.info("✅ No missing values to impute!")
        return df
    
    df_work = df.copy()
    
    # Store original dtypes and categorical mappings
    original_dtypes = df.dtypes.to_dict()
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Encode categorical variables
    for col in categorical_cols:
        if df_work[col].notna().sum() > 0:  # Only encode if there are non-null values
            le = LabelEncoder()
            # Fit on non-null values
            non_null_mask = df_work[col].notna()
            le.fit(df_work.loc[non_null_mask, col])
            
            # Transform non-null values
            df_work.loc[non_null_mask, col] = le.transform(df_work.loc[non_null_mask, col])
            df_work[col] = df_work[col].astype(float)  # Convert to float for imputation
            
            label_encoders[col] = le
    
    # Apply Iterative Imputer to all columns
    st.write("🤖 **Applying advanced iterative imputation...**")
    
    imputer = IterativeImputer(
        max_iter=max_iter,
        random_state=random_state,
        verbose=0
    )
    
    # Impute all data
    imputed_data = imputer.fit_transform(df_work)
    df_imputed = pd.DataFrame(imputed_data, columns=df.columns, index=df.index)
    
    # Decode categorical variables back to original format
    for col in categorical_cols:
        if col in label_encoders:
            # Round to nearest integer and clip to valid range
            encoded_values = np.clip(np.round(df_imputed[col]).astype(int), 
                                   0, len(label_encoders[col].classes_) - 1)
            
            # Inverse transform
            df_imputed[col] = label_encoders[col].inverse_transform(encoded_values)
    
    # Restore original data types where possible
    for col, dtype in original_dtypes.items():
        try:
            if dtype in ['int64', 'int32']:
                df_imputed[col] = df_imputed[col].round().astype(dtype)
            elif col not in categorical_cols:
                df_imputed[col] = df_imputed[col].astype(dtype)
        except:
            pass  # Keep as float if conversion fails
    
    st.success("✅ Advanced iterative imputation completed!")
    return df_imputed

# --- Visualize missing values before and after ---
def visualize_missing(df, title="Missing Values Heatmap"):
    """
    Show a heatmap of missing values
    """
    if df.isnull().sum().sum() == 0:
        st.info("✅ No missing values detected.")
        return
    
    fig, ax = plt.subplots(figsize=(max(10, 0.5*len(df.columns)), 6))
    sns.heatmap(df.isnull(), cbar=True, cmap="RdYlBu_r", yticklabels=False, ax=ax)
    ax.set_title(title, fontweight="bold", fontsize=14)
    st.pyplot(fig)
    plt.close()

# --- Compare before and after imputation ---
def compare_imputation(df_original, df_imputed):
    """
    Compare missing values before and after imputation
    """
    st.subheader("📊 Imputation Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Before Imputation:**")
        visualize_missing(df_original, "Before Imputation")
        
    with col2:
        st.write("**After Imputation:**")
        visualize_missing(df_imputed, "After Imputation")
    
    # Summary statistics
    missing_before = df_original.isnull().sum().sum()
    missing_after = df_imputed.isnull().sum().sum()
    
    st.metric("Total Missing Values Removed", 
              missing_before - missing_after, 
              delta=f"-{missing_before - missing_after}")

# --- Get imputation statistics ---
def get_imputation_stats(df_original, df_imputed):
    """
    Get detailed statistics about the imputation process
    """
    stats = []
    
    for col in df_original.columns:
        missing_before = df_original[col].isnull().sum()
        missing_after = df_imputed[col].isnull().sum()
        
        if missing_before > 0:
            stats.append({
                'Column': col,
                'Missing Before': missing_before,
                'Missing After': missing_after,
                'Imputed Values': missing_before - missing_after,
                'Missing %': round((missing_before / len(df_original)) * 100, 2)
            })
    
    if stats:
        stats_df = pd.DataFrame(stats)
        st.subheader("📈 Detailed Imputation Statistics")
        st.dataframe(stats_df, use_container_width=True)
        return stats_df
    
    return pd.DataFrame()