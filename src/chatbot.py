# chatbot.py
import google.generativeai as genai
import os
from utils.secrets import load_api_key

def initialize_gemini():
    """Initializes the Gemini model using an environment variable."""
    try:
        # Use environment variable or secrets file
        API_KEY = load_api_key('GEMINI_API_KEY') or os.environ.get('GEMINI_API_KEY')
        if not API_KEY:
            print("ERROR: GEMINI_API_KEY environment variable not set.")
            return None

        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        print(f"Error initializing Gemini API: {e}")
        return None

def generate_chat_response(model, message, chat_history):
    """
    Generates a response from the Gemini model, instructing it to act as a
    Tamil-speaking agricultural assistant.
    """
    if not model:
        return "மன்னிக்கவும், எனது செயற்கை நுண்ணறிவு அமைப்பு தற்போது செயலிழந்துள்ளது. தயவுசெய்து மீண்டும் முயற்சிக்கவும்."

    # This is the crucial prompt that defines the chatbot's persona and language.
    prompt = f"""
    You are a friendly and helpful agricultural assistant for a farmer in Tamil Nadu, India.
    Your primary language for conversation is Tamil.
    Your name is "உழவன் நண்பன்" (Uzhavan Nanban).
    You must always respond in Tamil, even if the user asks in English.
    Your goal is to provide practical, simple, and encouraging advice on plant diseases, farming techniques, and weather.
    Keep your answers concise and easy to understand for a farmer.
    
    Here is the conversation history:
    {chat_history}
    
    Here is the user's latest message:
    "{message}"

    Your response in Tamil:
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating response from Gemini: {e}")
        return "மன்னிக்கவும், ஒரு பிழை ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்."

