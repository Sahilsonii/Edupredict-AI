import google.generativeai as genai

genai.configure(api_key="AIzaSyB6qPn9DppYiRy0I04dQ--bkNsg1CczlV0")

print("Available models:")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"- {model.name}")
