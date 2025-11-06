# list_gemini_models.py
import google.generativeai as genai
from utils.secrets import load_api_key

API_KEY = load_api_key('GEMINI_API_KEY')
if not API_KEY:
    raise SystemExit("GEMINI_API_KEY not found (env / .env / secrets.json).")

genai.configure(api_key=API_KEY)

try:
    models = genai.list_models()
    print("Available models:")
    for m in models:
        # m can be dict-like; print the raw object so you can inspect fields
        print(m)
except Exception as e:
    print("Error listing models:", e)