import google.generativeai as genai
import json
from pathlib import Path
import sys

# Configure stdout to handle utf-8
sys.stdout.reconfigure(encoding='utf-8')

def load_api_key():
    secrets_file = Path("secrets.json")
    if secrets_file.exists():
        with open(secrets_file, 'r') as f:
            secrets = json.load(f)
            return secrets.get('GEMINI_API_KEY')
    return None

api_key = load_api_key()
if not api_key:
    with open("available_models.txt", "w") as f:
        f.write("API Key not found in secrets.json")
    exit(1)

genai.configure(api_key=api_key)

try:
    with open("available_models.txt", "w", encoding='utf-8') as f:
        f.write("Listing available models:\n")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                f.write(f"{m.name}\n")
    print("Models written to available_models.txt")
except Exception as e:
    with open("available_models.txt", "w") as f:
        f.write(f"Error listing models: {e}")
    print(f"Error: {e}")
