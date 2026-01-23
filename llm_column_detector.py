# llm_column_detector.py - LLM-Powered Automatic Column Categorization
# Works with ANY CSV - no keywords needed!

import pandas as pd
from typing import Dict, List, Any
import json
import google.generativeai as genai

class LLMColumnDetector:
    """
    Uses LLM to automatically understand and categorize CSV columns.
    Works with ANY dataset - educational, financial, medical, retail, etc.
    """
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemma-3-27b-it')
    
    def analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Automatically categorize all columns using LLM.
        Returns: Column categories, data types, visualization suggestions
        """
        
        column_info = self._extract_column_info(df)
        prompt = self._create_analysis_prompt(column_info)
        
        response = self.model.generate_content(prompt)
        analysis = self._parse_llm_response(response.text)
        
        return analysis
    
    def _extract_column_info(self, df: pd.DataFrame) -> List[Dict]:
        """Extract relevant information about each column"""
        column_info = []
        
        for col in df.columns:
            # Convert sample values to native Python types
            sample_values = df[col].dropna().head(5).tolist()
            sample_values = [int(x) if isinstance(x, (pd.Int64Dtype, type(pd.NA))) or str(type(x)).startswith("<class 'numpy.int") else 
                           float(x) if str(type(x)).startswith("<class 'numpy.float") else 
                           str(x) for x in sample_values]
            
            info = {
                'name': str(col),
                'dtype': str(df[col].dtype),
                'sample_values': sample_values,
                'unique_count': int(df[col].nunique()),
                'null_count': int(df[col].isnull().sum()),
                'total_rows': int(len(df))
            }
            column_info.append(info)
        
        return column_info
    
    def _create_analysis_prompt(self, column_info: List[Dict]) -> str:
        """Create prompt for LLM to analyze columns"""
        
        prompt = f"""You are a data analysis expert. Analyze these CSV columns and categorize them.

COLUMNS TO ANALYZE:
{json.dumps(column_info, indent=2)}

TASK: For each column, determine:
1. **Semantic Type**: What does this column represent?
   - Examples: student_id, enrollment_count, department_name, date, score, etc.

2. **Category**: What category does it belong to?
   - time_based: dates, years, periods, timestamps
   - identifier: IDs, codes, unique identifiers
   - metric: counts, scores, amounts, measurements
   - categorical: categories, labels, groups
   - text: descriptions, names, free text
   - geographic: locations, addresses, coordinates

3. **Visualization Type**: Best chart for this column
   - line_chart: for time series
   - bar_chart: for categorical comparisons
   - pie_chart: for proportions
   - scatter_plot: for correlations
   - histogram: for distributions
   - heatmap: for matrices

4. **Relationships**: Which columns should be analyzed together?
   - Example: "enrollment_count" over "year" → line chart

RESPOND IN THIS EXACT JSON FORMAT:
{{
  "columns": [
    {{
      "name": "column_name",
      "semantic_type": "what_it_represents",
      "category": "time_based|identifier|metric|categorical|text|geographic",
      "visualization_type": "line_chart|bar_chart|pie_chart|scatter_plot|histogram|heatmap",
      "description": "brief description"
    }}
  ],
  "suggested_analyses": [
    {{
      "type": "trend_analysis|comparison|distribution|correlation",
      "columns": ["col1", "col2"],
      "visualization": "chart_type",
      "description": "what this analysis shows"
    }}
  ],
  "domain": "education|finance|healthcare|retail|generic",
  "confidence": 0-100
}}

Respond ONLY with valid JSON, no other text."""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            
            analysis = json.loads(json_str)
            return analysis
        except Exception as e:
            # Fallback to basic analysis
            return {
                "columns": [],
                "suggested_analyses": [],
                "domain": "generic",
                "confidence": 0,
                "error": str(e)
            }
    
    def get_visualization_config(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert LLM analysis into visualization configuration.
        Returns ready-to-use config for Plotly/PyGwalker.
        """
        
        viz_config = {
            'time_series': [],
            'comparisons': [],
            'distributions': [],
            'correlations': []
        }
        
        # Extract columns by category
        time_cols = [c['name'] for c in analysis['columns'] if c['category'] == 'time_based']
        metric_cols = [c['name'] for c in analysis['columns'] if c['category'] == 'metric']
        categorical_cols = [c['name'] for c in analysis['columns'] if c['category'] == 'categorical']
        
        # Generate visualization configs
        for suggestion in analysis.get('suggested_analyses', []):
            if suggestion['type'] == 'trend_analysis':
                viz_config['time_series'].append({
                    'x': suggestion['columns'][0],
                    'y': suggestion['columns'][1],
                    'chart': suggestion['visualization'],
                    'title': suggestion['description']
                })
            elif suggestion['type'] == 'comparison':
                viz_config['comparisons'].append({
                    'category': suggestion['columns'][0],
                    'value': suggestion['columns'][1],
                    'chart': suggestion['visualization'],
                    'title': suggestion['description']
                })
        
        return viz_config


# Example usage function
def auto_analyze_csv(df: pd.DataFrame, api_key: str) -> Dict[str, Any]:
    """
    One-line function to automatically analyze ANY CSV.
    
    Usage:
        df = pd.read_csv("any_file.csv")
        analysis = auto_analyze_csv(df, api_key)
        
        # Now you have:
        # - Column categories
        # - Visualization suggestions
        # - Relationship insights
    """
    detector = LLMColumnDetector(api_key)
    analysis = detector.analyze_columns(df)
    viz_config = detector.get_visualization_config(analysis)
    
    return {
        'analysis': analysis,
        'visualizations': viz_config,
        'domain': analysis.get('domain', 'generic'),
        'confidence': analysis.get('confidence', 0)
    }


# Integration with existing system
def enhance_predictions_with_llm(df: pd.DataFrame, api_key: str) -> Dict[str, Any]:
    """
    Enhance existing predictions.py with LLM intelligence.
    Combines keyword matching + LLM understanding.
    """
    
    # Step 1: Get LLM analysis
    llm_analysis = auto_analyze_csv(df, api_key)
    
    # Step 2: Use LLM results for smart processing
    time_cols = [c['name'] for c in llm_analysis['analysis']['columns'] 
                 if c['category'] == 'time_based']
    
    metric_cols = [c['name'] for c in llm_analysis['analysis']['columns'] 
                   if c['category'] == 'metric']
    
    # Step 3: Return enhanced configuration
    return {
        'llm_analysis': llm_analysis,
        'time_columns': time_cols,
        'metric_columns': metric_cols,
        'suggested_forecasts': [
            {'target': m, 'time': t} 
            for m in metric_cols 
            for t in time_cols
        ],
        'visualization_config': llm_analysis['visualizations']
    }
