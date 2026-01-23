# analyzer.py - CSV Analysis Logic

import pandas as pd
import re

def analyze_csv_structure(df: pd.DataFrame):
    """
    Analyzes the structure of a CSV DataFrame.
    Returns a dictionary characterizing the columns.
    """
    analysis = {
        'numeric_cols': list(df.select_dtypes(include=['int64', 'float64']).columns),
        'text_cols': list(df.select_dtypes(include=['object']).columns),
        'date_cols': [],
        'categorical_cols': [],
        'key_patterns': {}
    }
    for col in df.columns:
        if any(term in col.lower() for term in ['date', 'time', 'year', 'month', 'period']):
            analysis['date_cols'].append(col)
    for col in analysis['text_cols']:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.5 and df[col].nunique() < 100:
            analysis['categorical_cols'].append(col)
    for col in df.columns:
        col_lower = col.lower()
        sample_values = df[col].dropna().astype(str).head(10).tolist()
        analysis['key_patterns'][col] = {
            'column_name': col,
            'sample_values': sample_values,
            'unique_count': df[col].nunique(),
            'has_numbers': any(char.isdigit() for char in col_lower),
            'keywords': [word for word in col_lower.split('_') if len(word) > 2]
        }
    return analysis

def extract_question_keywords(question):
    """Extract meaningful keywords from user question"""
    question_lower = question.lower()
    
    patterns = {
        'count_words': ['how many', 'count', 'total', 'number of', 'sum'],
        'time_words': ['year', 'month', 'date', 'time', 'period', 'when'],
        'comparison_words': ['highest', 'lowest', 'most', 'least', 'compare', 'versus'],
        'grouping_words': ['by', 'group', 'category', 'type', 'kind'],
        'statistical_words': ['average', 'mean', 'median', 'min', 'max', 'range']
    }
    
    found_patterns = {}
    for pattern_type, words in patterns.items():
        found_patterns[pattern_type] = [word for word in words if word in question_lower]
    
    numbers = re.findall(r'\b\d{4}\b|\b\d+\.?\d*\b', question)
    quoted_terms = re.findall(r'"([^"]*)"', question) + re.findall(r"'([^']*)'", question)
    
    return {
        'patterns': found_patterns,
        'numbers': numbers,
        'quoted_terms': quoted_terms,
        'question_words': question_lower.split()
    }

def find_relevant_data_smart(df, question, csv_analysis):
    """Universal function to find relevant data based on ANY question and ANY CSV structure"""
    
    question_analysis = extract_question_keywords(question)
    relevant_data = df.copy()
    filters_applied = []
    relevance_score = {}
    
    # Score columns by relevance to question
    for col, col_info in csv_analysis['key_patterns'].items():
        score = 0
        col_lower = col.lower()
        
        # Check if column name matches question keywords
        for word in question_analysis['question_words']:
            if word in col_lower or word in ' '.join(col_info['keywords']):
                score += 10
        
        # Check if sample values match question terms
        for value in col_info['sample_values']:
            for word in question_analysis['question_words']:
                if word in str(value).lower():
                    score += 5
        
        # Bonus for specific quoted terms
        for term in question_analysis['quoted_terms']:
            if term.lower() in col_lower or any(term.lower() in str(val).lower() for val in col_info['sample_values']):
                score += 20
        
        relevance_score[col] = score
    
    # Filter by specific values mentioned in question
    for col in df.columns:
        col_data = df[col].astype(str)
        
        # Filter by numbers mentioned
        for num in question_analysis['numbers']:
            mask = col_data.str.contains(num, na=False, case=False)
            if mask.any():
                old_len = len(relevant_data)
                relevant_data = relevant_data[mask]
                if len(relevant_data) < old_len:
                    filters_applied.append(f"contains '{num}' in {col}")
                if len(relevant_data) < 50:
                    break
        
        # Filter by quoted terms
        for term in question_analysis['quoted_terms']:
            mask = col_data.str.contains(term, na=False, case=False)
            if mask.any():
                old_len = len(relevant_data)
                relevant_data = relevant_data[mask]
                if len(relevant_data) < old_len:
                    filters_applied.append(f"contains '{term}' in {col}")
    
    return relevant_data, filters_applied, relevance_score

def create_universal_context(df, question):
    """Create context for ANY CSV file based on the question asked"""
    
    csv_analysis = analyze_csv_structure(df)
    relevant_data, filters_applied, relevance_scores = find_relevant_data_smart(df, question, csv_analysis)
    
    context = []
    context.append(f"=== UNIVERSAL CSV ANALYSIS ===")
    context.append(f"Dataset: {len(df):,} rows, {len(df.columns)} columns")
    context.append(f"Columns: {', '.join(df.columns)}")
    
    # Add column analysis
    context.append(f"\n=== COLUMN ANALYSIS ===")
    for col in df.columns:
        col_type = "Numeric" if col in csv_analysis['numeric_cols'] else "Text"
        if col in csv_analysis['categorical_cols']:
            col_type += " (Categorical)"
        if col in csv_analysis['date_cols']:
            col_type += " (Date/Time)"
        
        context.append(f"'{col}' ({col_type}): {df[col].nunique()} unique values")
        
        # Add sample values
        if df[col].dtype in ['int64', 'float64']:
            context.append(f"  Range: {df[col].min()} to {df[col].max()}")
        else:
            sample_vals = df[col].value_counts().head(5)
            context.append(f"  Top values: {dict(sample_vals)}")
    
    # Add filtering information
    if filters_applied:
        context.append(f"\n=== SMART FILTERING APPLIED ===")
        context.append(f"Filters: {', '.join(filters_applied)}")
        context.append(f"Filtered from {len(df):,} to {len(relevant_data):,} rows")
    
    # Add the actual data
    if len(relevant_data) <= 500:
        context.append(f"\n=== COMPLETE RELEVANT DATA ({len(relevant_data)} rows) ===")
        context.append(relevant_data.to_string(index=False))
    elif len(relevant_data) <= 2000:
        context.append(f"\n=== RELEVANT DATA SAMPLE ({len(relevant_data)} total rows) ===")
        context.append(f"First 200 rows:")
        context.append(relevant_data.head(200).to_string(index=False))
        context.append(f"\nLast 100 rows:")
        context.append(relevant_data.tail(100).to_string(index=False))
    else:
        context.append(f"\n=== STRATEGIC SAMPLE FROM {len(relevant_data)} RELEVANT ROWS ===")
        context.append(f"First 150 rows:")
        context.append(relevant_data.head(150).to_string(index=False))
        context.append(f"\nRandom sample (100 rows):")
        sample = relevant_data.sample(n=min(100, len(relevant_data)-150), random_state=42)
        context.append(sample.to_string(index=False))
        context.append(f"\nLast 50 rows:")
        context.append(relevant_data.tail(50).to_string(index=False))
        
        # Add summary statistics for large datasets
        context.append(f"\n=== SUMMARY STATISTICS ===")
        for col in csv_analysis['categorical_cols']:
            if col in relevant_data.columns:
                counts = relevant_data[col].value_counts()
                context.append(f"{col} counts: {dict(counts.head(10))}")
    
    return "\n".join(context)
