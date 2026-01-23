# 🤖 LLM Column Detector - Standalone App

## Quick Start

### 1. Install Dependencies
```bash
pip install streamlit pandas langchain-google-genai
```

### 2. Add API Key
Create `secrets.json` in the same folder:
```json
{
  "GEMINI_API_KEY": "your_gemini_api_key_here"
}
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Run the App
```bash
streamlit run llm_detector_app.py
```

### 4. Upload CSV
- Click "Browse files"
- Select any CSV file
- Click "Analyze with AI"
- Get instant results!

---

## What You Get

### ✅ Automatic Column Analysis
- **Semantic Type**: What each column represents
- **Category**: time_based, metric, categorical, identifier, etc.
- **Best Visualization**: Recommended chart type
- **Description**: AI-generated explanation

### ✅ Domain Detection
- Automatically identifies dataset type
- Education, Finance, Healthcare, Retail, Generic
- Confidence score (0-100%)

### ✅ Smart Suggestions
- Recommended analyses
- Column relationships
- Visualization types

### ✅ Export Options
- Download as JSON
- Download as CSV
- Copy raw response

---

## Example Output

```json
{
  "columns": [
    {
      "name": "Year",
      "semantic_type": "academic_year",
      "category": "time_based",
      "visualization_type": "line_chart",
      "description": "Academic year for tracking"
    },
    {
      "name": "Student_Count",
      "semantic_type": "enrollment_metric",
      "category": "metric",
      "visualization_type": "line_chart",
      "description": "Number of enrolled students"
    }
  ],
  "suggested_analyses": [
    {
      "type": "trend_analysis",
      "columns": ["Year", "Student_Count"],
      "visualization": "line_chart",
      "description": "Student enrollment trend over time"
    }
  ],
  "domain": "education",
  "confidence": 95
}
```

---

## Features

✅ Works with **ANY CSV** structure  
✅ No manual configuration needed  
✅ AI-powered understanding  
✅ Fast analysis (<5 seconds)  
✅ Export results  
✅ Beautiful UI  

---

## Cost

- **Per Analysis**: ~$0.001 (less than 1 cent)
- **1000 analyses**: ~$1-2/month
- **Negligible cost** for massive value!

---

## Troubleshooting

### Error: "API key not found"
- Make sure `secrets.json` exists in the same folder
- Check the API key is valid
- Format: `{"GEMINI_API_KEY": "your_key"}`

### Error: "Module not found"
- Run: `pip install streamlit pandas langchain-google-genai`

### Slow analysis
- Normal for first run (model loading)
- Subsequent runs are cached and faster

---

## Files Needed

```
EduPredictv2/
├── llm_column_detector.py    # Core detection logic
├── llm_detector_app.py        # Streamlit UI (run this)
└── secrets.json               # Your API key
```

---

## Next Steps

After analyzing your CSV:
1. Review column categories
2. Check suggested analyses
3. Export results
4. Use insights for visualization
5. Integrate with main app (optional)

---

**Enjoy automatic column detection! 🚀**
