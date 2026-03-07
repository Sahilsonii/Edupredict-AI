# planning.py - AI-Powered Planning Studio with Gemma Integration

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import google.generativeai as genai
import json
import streamlit as st

class PlanningStudio:
    """
    AI-powered planning and decision support system
    Features: Target tracking, What-if scenarios, Anomaly detection, Recommendations
    """
    
    def __init__(self, api_key: str):
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemma-3-27b-it')
        else:
            self.model = None
    
    # ===== CORE PLANNING FUNCTIONS =====
    
    def calculate_target_vs_actual(
        self, 
        df: pd.DataFrame, 
        metric: str, 
        time_col: str,
        target_mode: str = "growth",
        growth_rate: float = 0.05,
        manual_target: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculate target vs actual vs forecast"""
        
        # Get historical data
        historical = df.groupby(time_col)[metric].sum().sort_index()
        
        if len(historical) < 2:
            return {"error": "Insufficient data"}
        
        # Calculate targets
        targets = {}
        for period in historical.index:
            if target_mode == "growth":
                if period == historical.index[0]:
                    targets[period] = historical.iloc[0]
                else:
                    prev_idx = list(historical.index).index(period) - 1
                    targets[period] = historical.iloc[prev_idx] * (1 + growth_rate)
            else:
                targets[period] = manual_target if manual_target else historical.mean()
        
        # Calculate gaps
        gaps = {period: historical[period] - targets[period] for period in historical.index}
        
        return {
            "historical": historical.to_dict(),
            "targets": targets,
            "gaps": gaps,
            "achievement_rate": {
                period: (historical[period] / targets[period] * 100) if targets[period] != 0 else 0
                for period in historical.index
            }
        }
    
    def detect_anomalies(
        self, 
        df: pd.DataFrame, 
        metric: str, 
        time_col: str,
        threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using statistical methods"""
        
        series = df.groupby(time_col)[metric].sum().sort_index()
        
        if len(series) < 3:
            return []
        
        # Calculate rolling statistics
        mean = series.mean()
        std = series.std()
        
        anomalies = []
        for period, value in series.items():
            z_score = abs((value - mean) / std) if std != 0 else 0
            
            if z_score > threshold:
                # Calculate change from previous period
                idx = list(series.index).index(period)
                if idx > 0:
                    prev_value = series.iloc[idx - 1]
                    change_pct = ((value - prev_value) / prev_value * 100) if prev_value != 0 else 0
                else:
                    change_pct = 0
                
                anomalies.append({
                    "period": period,
                    "value": value,
                    "z_score": z_score,
                    "change_pct": change_pct,
                    "severity": "High" if z_score > 3 else "Medium"
                })
        
        return anomalies
    
    def simulate_scenario(
        self,
        df: pd.DataFrame,
        metric: str,
        time_col: str,
        adjustments: Dict[str, float]
    ) -> Dict[str, Any]:
        """Simulate what-if scenarios"""
        
        baseline = df.groupby(time_col)[metric].sum().sort_index()
        
        # Apply adjustments
        simulated = baseline.copy()
        for adj_metric, multiplier in adjustments.items():
            if adj_metric == metric:
                simulated = simulated * multiplier
        
        # Calculate impact
        total_baseline = baseline.sum()
        total_simulated = simulated.sum()
        impact = total_simulated - total_baseline
        impact_pct = (impact / total_baseline * 100) if total_baseline != 0 else 0
        
        return {
            "baseline": baseline.to_dict(),
            "simulated": simulated.to_dict(),
            "impact": impact,
            "impact_pct": impact_pct
        }
    
    # ===== GEMMA-POWERED AI FEATURES =====
    
    def parse_scenario_from_text(
        self,
        user_input: str,
        available_metrics: List[str]
    ) -> Dict[str, float]:
        """Parse natural language scenario into parameters using Gemma"""
        
        if not self.model:
            return {"error": "API key not configured"}
        
        prompt = f"""
Parse this scenario request into adjustment parameters.

User request: "{user_input}"

Available metrics: {available_metrics}

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{{
    "metric_name": multiplier_value
}}

Examples:
- "increase enrollment by 10%" → {{"enrollment": 1.10}}
- "decrease dropout by 5%" → {{"dropout": 0.95}}
- "boost revenue by 15%" → {{"revenue": 1.15}}

If metric not in available list, use closest match.
"""
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip().replace('```json', '').replace('```', '').strip()
            parsed = json.loads(text)
            return parsed
        except Exception as e:
            return {"error": f"Parsing failed: {str(e)}"}
    
    def explain_anomaly(
        self,
        df: pd.DataFrame,
        metric: str,
        anomaly_info: Dict[str, Any],
        dimensions: List[str]
    ) -> str:
        """Use Gemma to explain why anomaly occurred"""
        
        if not self.model:
            return "AI explanation not available (API key not configured)"
        
        period = anomaly_info['period']
        change_pct = anomaly_info['change_pct']
        
        # Get segment breakdown for that period
        period_data = df[df.iloc[:, 0] == period] if len(df) > 0 else df
        
        segment_summary = {}
        for dim in dimensions[:3]:  # Limit to 3 dimensions
            if dim in df.columns:
                segment_summary[dim] = period_data.groupby(dim)[metric].sum().to_dict()
        
        prompt = f"""
You are an academic data analyst. Explain this anomaly:

Metric: {metric}
Period: {period}
Change: {change_pct:.1f}%
Severity: {anomaly_info['severity']}

Segment breakdown:
{json.dumps(segment_summary, indent=2)}

Provide:
1. Root cause (2-3 sentences)
2. Top 2-3 contributing segments
3. 2 recommended actions

Keep it concise and actionable.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Explanation failed: {str(e)}"
    
    def generate_executive_summary(
        self,
        target_analysis: Dict[str, Any],
        anomalies: List[Dict[str, Any]],
        metric: str
    ) -> str:
        """Generate executive summary using Gemma"""
        
        if not self.model:
            return "Executive summary not available (API key not configured)"
        
        prompt = f"""
Create a concise executive summary for academic leadership.

Metric: {metric}

Performance:
- Historical data: {len(target_analysis.get('historical', {}))} periods
- Average achievement rate: {np.mean(list(target_analysis.get('achievement_rate', {}).values())):.1f}%

Anomalies detected: {len(anomalies)}
{json.dumps(anomalies[:3], indent=2) if anomalies else "None"}

Format:
1. Performance Overview (2-3 sentences)
2. Key Risks (2 points if any)
3. Recommendations (2-3 actions)

Keep it executive-level: clear, concise, actionable.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def get_recommendations(
        self,
        df: pd.DataFrame,
        metric: str,
        target_analysis: Dict[str, Any],
        anomalies: List[Dict[str, Any]]
    ) -> str:
        """Get AI-powered recommendations using Gemma"""
        
        if not self.model:
            return "Recommendations not available (API key not configured)"
        
        # Calculate key statistics
        gaps = target_analysis.get('gaps', {})
        avg_gap = np.mean(list(gaps.values())) if gaps else 0
        achievement_rate = target_analysis.get('achievement_rate', {})
        avg_achievement = np.mean(list(achievement_rate.values())) if achievement_rate else 100
        
        prompt = f"""
You are an academic planning advisor. Provide actionable recommendations.

Current Situation:
- Metric: {metric}
- Average gap from target: {avg_gap:.0f}
- Average achievement rate: {avg_achievement:.1f}%
- Anomalies: {len(anomalies)}

Provide 3-4 specific recommendations with:
- Action description
- Expected impact (quantified if possible)
- Implementation difficulty (Low/Medium/High)
- Timeline (Short-term/Medium-term/Long-term)

Format as numbered list. Be specific and actionable.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Recommendations failed: {str(e)}"
