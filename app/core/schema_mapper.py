# schema_mapper.py - LLM-powered Schema Standardization Layer

import pandas as pd
import json
import hashlib
import os
import streamlit as st
from typing import Dict, Any, List, Optional
import google.generativeai as genai

# Canonical Internal Schema
# This defines how we WANT the data to look internally.
INTERNAL_SCHEMA = {
    "metrics": {
        "description": "Numerical values to analyze or forecast",
        "examples": ["enrollment", "graduates", "dropout_rate", "revenue", "score"]
    },
    "dimensions": {
        "description": "Categorical values to group by",
        "examples": ["department", "gender", "ethnicity", "course", "region"]
    },
    "time": {
        "description": "Time periods",
        "examples": ["year", "academic_year", "date", "semester"]
    }
}

class SchemaMapper:
    """
    Standardizes diverse CSV headers into a canonical internal schema using LLM.
    Acts as a translation layer: Raw CSV -> Standardized Metadata
    """
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API Key is required for SchemaMapper")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemma-3-27b-it')
        
    def _compute_file_hash(self, df: pd.DataFrame) -> str:
        """Create a hash of columns to cache results"""
        col_str = ",".join(sorted(df.columns.astype(str)))
        return hashlib.md5(col_str.encode()).hexdigest()

    @st.cache_data(show_spinner=False)
    def get_standardized_map(_self, header_sample: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Uses LLM to map raw columns to our internal schema.
        Cached by Streamlit based on input arguments.
        """
        prompt = f"""
        You are an Expert Academic Data Engineer.
        
        INTERNAL SCHEMA DEFINITION:
        - metrics: {INTERNAL_SCHEMA['metrics']['description']} (e.g. {INTERNAL_SCHEMA['metrics']['examples']})
        - dimensions: {INTERNAL_SCHEMA['dimensions']['description']} (e.g. {INTERNAL_SCHEMA['dimensions']['examples']})
        - time: {INTERNAL_SCHEMA['time']['description']} (e.g. {INTERNAL_SCHEMA['time']['examples']})
        
        RAW CSV COLUMNS & SAMPLES:
        {json.dumps(header_sample, indent=2)}
        
        TASK:
        1. VALIDATE: Is this dataset related to the ACADEMIC/EDUCATION domain? (Students, Universities, Grades, Enrollment, Research, Schools, etc.)
        2. IF NOT ACADEMIC: Return "is_academic": false.
        3. IF ACADEMIC: Map raw columns to semantically standardized names.
        
        RETURN JSON ONLY:
        {{
            "is_academic": true/false,
            "rejection_reason": "Reason if not academic, else null",
            "mapping": {{
                "raw_column_name_1": {{
                    "role": "metric|dimension|time|ignore",
                    "canonical_name": "standardized_snake_case_name",
                    "description": "short description"
                }},
                ...
            }}
        }}
        """
        
        try:
            response = _self.model.generate_content(prompt)
            # Basic cleanup if model adds markdown backticks
            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except Exception as e:
            return {"error": str(e)}

    def standardize(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main entry point. Returns the mapping and specific column sets.
        """
        # Prepare small sample for LLM (privacy safe: only headers + 3 values)
        sample = {}
        for col in df.columns:
            sample[str(col)] = df[col].dropna().head(3).astype(str).tolist()
            
        # Get mapping (cached)
        llm_response = self.get_standardized_map(sample)
        
        if "error" in llm_response:
             # Fallback to heuristics if LLM fails (network/api error)
             return self._heuristic_fallback(df)

        # STRICT ACADEMIC CHECK
        if llm_response.get("is_academic") is False:
            return {
                "error": "Non-Academic Data Detected",
                "message": llm_response.get("rejection_reason", "This dataset does not appear to be educational/academic.")
            }
            
        mapping = llm_response.get("mapping", {})

        # Organize results
        result = {
            "full_mapping": mapping,
            "metrics": [],
            "dimensions": [],
            "time_col": None
        }
        
        for raw_col, meta in mapping.items():
            if raw_col not in df.columns: continue
            
            role = meta.get("role")
            if role == "metric":
                result["metrics"].append(raw_col)
            elif role == "dimension":
                result["dimensions"].append(raw_col)
            elif role == "time":
                # Only take the first identified time column as primary for now
                if result["time_col"] is None:
                    result["time_col"] = raw_col
                    
        return result

    def _heuristic_fallback(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Basic fallback if LLM fails"""
        numeric = df.select_dtypes(include=['number']).columns.tolist()
        objects = df.select_dtypes(include=['object']).columns.tolist()
        
        # Simple guess: 'year' or 'date' in name -> time
        time_col = None
        for col in df.columns:
            if 'year' in col.lower() or 'date' in col.lower():
                time_col = col
                break
        
        if time_col and time_col in numeric:
            numeric.remove(time_col)
            
        return {
            "full_mapping": {}, # Empty map indicates fallback
            "metrics": numeric,
            "dimensions": objects,
            "time_col": time_col
        }
