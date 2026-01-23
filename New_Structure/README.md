# EduPredict AI (Refactored)

This is the refactored version of EduPredictv2 with a modular architecture.

## Structure

- **main.py**: Entry point.
- **app/**: Core application code.
  - **core/**: Business logic (ML, Analysis, Data handling).
  - **ui/**: Streamlit UI components.
  - **utils/**: Hepler functions.
- **data/**: Data storage.
- **config/**: Configuration files.

## How to Run

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   streamlit run main.py
   ```

## Configuration

- API Keys are loaded from `.env` file.
