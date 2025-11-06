# app.py

# IMPORT render_template to serve your HTML page
from flask import Flask, request, jsonify, render_template 
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models # Import models for Vision Transformer
from PIL import Image
import re
import json
import os
import requests
import torch.serialization
import google.generativeai as genai
from utils.secrets import load_api_key
import time
import threading
from chatbot import initialize_gemini, generate_chat_response

# ----------------- Flask App -----------------
app = Flask(__name__)
CORS(app)

# ----------------- Paths -----------------
model_path = "ml_models/plant_disease_model.pth"
classes_path = 'ml_models/classes.json'


# ----------------- Gemini API Setup (lazy, robust) -----------------
# We'll attempt to initialize Gemini when needed. This helps when the Flask
# reloader or process didn't have the env set at import time.
model_gemini = None
gemini_model_name = None
# Simple in-memory cache for care plans to avoid repeated LLM calls for the same (disease, risk)
CARE_PLAN_CACHE = {}

# Chatbot model and sessions (uses src/chatbot.py without modification)
chat_model = None
CHAT_SESSIONS = {}


def ensure_gemini_initialized():
    """Attempt to configure the Google generative API and initialize a model.
    Returns True if a model is available, False otherwise.
    This function is safe to call multiple times.
    """
    global model_gemini, gemini_model_name
    # If already initialized, skip
    if model_gemini is not None:
        return True
    start_t = time.perf_counter()

    API_KEY = load_api_key('GEMINI_API_KEY')
    if not API_KEY:
        print("Warning: GEMINI_API_KEY not found in environment/.env/secrets.json. Gemini features disabled.")
        return False

    try:
        genai.configure(api_key=API_KEY)
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        return False

    preferred_model = load_api_key('GEMINI_MODEL')
    if preferred_model:
        try:
            model_gemini = genai.GenerativeModel(preferred_model)
            gemini_model_name = preferred_model
            print(f"Initialized Gemini model from GEMINI_MODEL: {preferred_model}")
            return True
        except Exception as e:
            print(f"Could not initialize GEMINI_MODEL='{preferred_model}': {e}")
            model_gemini = None

    # Auto-select a model that supports generateContent
    try:
        available_models = genai.list_models() or []
        gen_models = [m for m in available_models if getattr(m, 'supported_generation_methods', []) and 'generateContent' in m.supported_generation_methods]
        # prefer known families if present
        priority = ['2.5', '2.0', 'flash', 'pro', 'gemini']
        chosen = None
        for p in priority:
            for m in gen_models:
                name = getattr(m, 'name', str(m))
                if p in name:
                    chosen = name
                    break
            if chosen:
                break
        if not chosen and gen_models:
            chosen = getattr(gen_models[0], 'name', str(gen_models[0]))

        if chosen:
            try:
                model_gemini = genai.GenerativeModel(chosen)
                gemini_model_name = chosen
                print(f"Initialized Gemini model: {chosen}")
                return True
            except Exception as e:
                print(f"Failed to initialize auto-chosen model '{chosen}': {e}")
                model_gemini = None
    except Exception as e:
        print(f"Error while listing Gemini models: {e}")
        model_gemini = None

    print("Gemini not initialized. Generation disabled.")
    print(f"ensure_gemini_initialized took {time.perf_counter() - start_t:.2f}s")
    return False

# ----------------- Load Class Names -----------------
with open(classes_path, "r") as f:
    classes = json.load(f)

# ----------------- Model Architecture (Disease Model) -----------------
class CNN_NeuralNet(nn.Module):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(4))
        self.res1 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(4))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(4))
        self.res2 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_diseases))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# --- Load Models ---
torch.serialization.add_safe_globals([CNN_NeuralNet])
disease_model = torch.load(model_path, map_location="cpu", weights_only=False)
disease_model.eval()

# --- MODIFICATION: Load Vision Transformer (ViT) as the "Specialist Guard" ---
weights = models.ViT_B_16_Weights.IMAGENET1K_V1
guard_model = models.vit_b_16(weights=weights)
guard_model.eval()

# --- Get the list of all 1000 class names from ImageNet ---
imagenet_classes = weights.meta["categories"]

# --- Create our "master list" of plant-related classes ---
PLANT_RELATED_CLASSES = {
    # Generic ImageNet classes that are plant-related
    'maize', 'corn', 'oak tree', 'maple tree', 'daisy', 'yellow lady\'s slipper',
    'pot', 'flowerpot', 'head cabbage', 'broccoli', 'cauliflower', 'zucchini', 
    'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber', 'artichoke', 
    'bell pepper', 'cardoon', 'mushroom', 'Granny Smith', 'strawberry', 'orange',
    'lemon', 'fig', 'pineapple', 'banana', 'jackfruit', 'custard apple', 'pomegranate',
    'acorn', 'hip', 'rapeseed', 'leaf beetle', 'lacewing', 'earthstar', 'stinkhorn',
    'velvet foot',
    # Classes extracted specifically from your disease list
    'apple', 'blueberry', 'cherry', 'grape', 'peach', 'pepper', 'potato', 
    'raspberry', 'soybean', 'squash', 'tomato'
}


# Confidence threshold for the guard model's prediction.
GUARD_CONFIDENCE_THRESHOLD = 0.10 # Lowered threshold as it's one of 1000 classes


# --- Define separate transforms for each model ---
disease_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# --- MODIFICATION: Get the official transforms for the ViT model ---
guard_transform = weights.transforms()


# ----------------- Root Route: show login first -----------------
@app.route("/")
def index():
    """Serve the farmer login as the default landing page so users must sign in first."""
    return render_template("farmer_login.html")


# Explicit route to serve the login page (keeps URLs intuitive)
@app.route('/farmer_login')
def farmer_login():
    return render_template('farmer_login.html')


# Dashboard (land details) route: user sees this after login
@app.route('/land_details')
def land_details():
    return render_template('land_details.html')


# Route to serve the disease detection/upload UI (invoked from dashboard)
@app.route('/disease_detection')
def disease_detection():
    return render_template('index.html')


@app.route('/chat')
def chat_ui():
    """Serve the chatbot UI page."""
    return render_template('chat.html')

# ----------------- Weather Risk Route -----------------
@app.route("/weather_risk", methods=["POST"])
def weather_risk():
    data = request.get_json()
    lat, lon = data.get('lat'), data.get('lon')
    if not lat or not lon:
        return jsonify({"error": "Latitude and Longitude are required."}), 400
    
    WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY" # Replace with your key
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        weather_data = response.json()
        temp, humidity = weather_data['main']['temp'], weather_data['main']['humidity']
        risk = "High" if humidity > 80 and 18 <= temp <= 28 else "Medium" if humidity > 70 and 15 <= temp <= 30 else "Low"
        return jsonify({"temperature": temp, "humidity": humidity, "risk": risk})
    except requests.exceptions.RequestException as e:
        print(f"Weather API Request Error: {e}")
        return jsonify({"error": "Could not fetch weather data."}), 500
    except KeyError:
        print("Weather API Error: Unexpected JSON structure")
        return jsonify({"error": "Could not parse weather data."}), 500

# ----------------- Prediction API (MODIFIED) -----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    risk = request.form.get("risk", "Not available")

    try:
        image = Image.open(file).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image format"}), 400

    # ===============================================================================
    # STEP 1: "Full Knowledge" Guard check using the original Vision Transformer
    # ===============================================================================
    img_tensor_guard = guard_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs_guard = guard_model(img_tensor_guard)
        probabilities_guard = torch.nn.functional.softmax(outputs_guard, dim=1)
        confidence_guard, predicted_idx_guard = torch.max(probabilities_guard, 1)
        
        # Get the name of the predicted class (e.g., 'daisy', 'car', 'pot')
        predicted_class_name = imagenet_classes[predicted_idx_guard.item()]

    # --- For debugging: print what the guard model sees ---
    print(f"Guard Model Prediction: '{predicted_class_name}' with confidence {confidence_guard.item():.2f}")

    # --- MODIFIED Robust check ---
    # Is the predicted class name in our master list of plant-related things?
    # AND is the confidence high enough?
    is_plant = predicted_class_name in PLANT_RELATED_CLASSES and confidence_guard.item() > GUARD_CONFIDENCE_THRESHOLD

    if not is_plant:
        return jsonify({
            "disease": "Not a Plant",
            "confidence": 0,
            "care_plan": f"The model identified this as a '{predicted_class_name.replace('_', ' ')}', which does not seem to be a plant. Please upload a clear image of a plant leaf."
        })

    # ===============================================================================
    # STEP 2: If it's a plant, proceed to disease classification
    # ===============================================================================
    img_t_disease = disease_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs_disease = disease_model(img_t_disease)
        probabilities = torch.nn.functional.softmax(outputs_disease, dim=1)
        confidence, predicted_disease = torch.max(probabilities, 1)

    predicted_class = classes[predicted_disease.item()]
    confidence_score = round(confidence.item() * 100, 2)
    clean_disease_name = predicted_class.replace('_', ' ').replace('___', ' ').title()
    DISEASE_CONFIDENCE_THRESHOLD = 90.0

    # Default structured care plan (ensures variable exists for all return paths)
    care_plan_structured = {
        'Immediate Actions': [],
        'Preventive Measures': [],
        'Long-Term Solutions': []
    }

    if confidence_score < DISEASE_CONFIDENCE_THRESHOLD:
        clean_disease_name = "Unrecognized Condition"
        care_plan = "The model is not confident enough for a reliable diagnosis. Please ensure the uploaded image is a clear, well-lit photo of the affected plant leaf."
    else:
        # Generate care plan (try Gemini, fall back to templates)
        try:
            care_plan, care_plan_structured = generate_care_plan(clean_disease_name, risk)
        except Exception as e:
            print(f"Error generating care plan: {e}")
            care_plan = "An error occurred while generating the care plan."
            care_plan_structured = {'Immediate Actions': [], 'Preventive Measures': [], 'Long-Term Solutions': []}

    return jsonify({
        "disease": clean_disease_name,
        "confidence": confidence_score,
        "care_plan": care_plan,
        "care_plan_structured": care_plan_structured
    })


# ----------------- Helper: generate care plan (Gemini or fallback) -----------------
def generate_care_plan(clean_disease_name, risk="Not available"):
    """Generate a care plan for a disease name. Returns (care_plan_text, care_plan_structured_dict)."""

    # Small local templates as safe fallback
    CARE_PLAN_TEMPLATES = {
        "Late Blight": {
            "Immediate Actions": [
                "Remove and destroy infected leaves and stems; do not compost.",
                "Avoid overhead watering; water at the base of plants.",
                "Apply a registered fungicide labeled for late blight if available and needed."
            ],
            "Preventive Measures": [
                "Use certified disease-free seed/potatoes.",
                "Practice crop rotation and avoid planting solanaceous crops in the same spot year-to-year.",
                "Improve air circulation by spacing plants properly."
            ],
            "Long-Term Solutions": [
                "Select resistant varieties when available.",
                "Monitor weather and apply protective sprays before high-risk periods.",
                "Maintain good field sanitation and remove volunteer plants."
            ]
        },
        "Early Blight": {
            "Immediate Actions": [
                "Remove affected foliage and destroy it.",
                "Ensure good drainage and avoid excess irrigation."
            ],
            "Preventive Measures": [
                "Rotate crops and avoid planting tomatoes/potatoes in the same place.",
                "Mulch to reduce soil splash onto leaves."
            ],
            "Long-Term Solutions": [
                "Use resistant cultivars where possible.",
                "Practice regular monitoring and sanitation."
            ]
        }
    }

    def generate_with_retries(model, prompt, max_attempts=3, base_delay=2):
        attempt = 0
        while attempt < max_attempts:
            try:
                return model.generate_content(prompt)
            except Exception as e:
                msg = str(e).lower()
                attempt += 1
                if 'quota' in msg or '429' in msg or 'rate-limit' in msg or 'rate limit' in msg:
                    delay = base_delay * (2 ** (attempt - 1))
                    print(f"Quota/rate limit error from Gemini (attempt {attempt}/{max_attempts}). Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                print(f"Non-retryable error from Gemini: {e}")
                raise
        return None

    prompt = f"""
    Generate a farmer-friendly care plan for a plant with "{clean_disease_name}".
    The current environmental risk is "{risk}".

    IMPORTANT: Use the following markdown structure EXACTLY. Do not add any introductory or concluding sentences outside of this structure.

    ### Immediate Actions
    - [Action 1]
    - [Action 2]
    - ...

    ### Preventive Measures
    - [Measure 1]
    - [Measure 2]
    - ...

    ### Long-Term Solutions
    - [Solution 1]
    - [Solution 2]
    - ...
    """

    # Check cache first to avoid repeated LLM calls
    cache_key = f"{clean_disease_name}|{risk}"
    cached = CARE_PLAN_CACHE.get(cache_key)
    if cached:
        cp, cps = cached
        print(f"Care plan cache hit for {cache_key}")
        return cp, dict(cps)

    # Try to initialize Gemini; if it's not available, use local templates
    if ensure_gemini_initialized():
        t_start = time.perf_counter()
        try:
            resp = generate_with_retries(model_gemini, prompt, max_attempts=3)
            if resp is None:
                tpl = CARE_PLAN_TEMPLATES.get(clean_disease_name)
                if tpl:
                    care_plan_structured = tpl
                    care_plan = "\n\n".join([
                        "### Immediate Actions\n- " + "\n- ".join(tpl['Immediate Actions']),
                        "### Preventive Measures\n- " + "\n- ".join(tpl['Preventive Measures']),
                        "### Long-Term Solutions\n- " + "\n- ".join(tpl['Long-Term Solutions'])
                    ])
                    try:
                        CARE_PLAN_CACHE[cache_key] = (care_plan, care_plan_structured)
                    except Exception:
                        pass
                    return care_plan, care_plan_structured
                else:
                    return "An error occurred while generating the care plan (quota exceeded).", {'Immediate Actions': [], 'Preventive Measures': [], 'Long-Term Solutions': []}
            # got a response
            care_plan = resp.text
            care_plan_structured = parse_care_plan_into_sections(care_plan)
            # cache result
            try:
                CARE_PLAN_CACHE[cache_key] = (care_plan, care_plan_structured)
            except Exception:
                pass
            print(f"Gemini generate took {time.perf_counter() - t_start:.2f}s for {clean_disease_name}")
            return care_plan, care_plan_structured
        except Exception as e:
            print(f"Error fetching care plan from Gemini: {e}")
            tpl = CARE_PLAN_TEMPLATES.get(clean_disease_name)
            if tpl:
                care_plan_structured = tpl
                care_plan = "\n\n".join([
                    "### Immediate Actions\n- " + "\n- ".join(tpl['Immediate Actions']),
                    "### Preventive Measures\n- " + "\n- ".join(tpl['Preventive Measures']),
                    "### Long-Term Solutions\n- " + "\n- ".join(tpl['Long-Term Solutions'])
                ])
                try:
                    CARE_PLAN_CACHE[cache_key] = (care_plan, care_plan_structured)
                except Exception:
                    pass
                return care_plan, care_plan_structured
            return "An error occurred while generating the care plan.", {'Immediate Actions': [], 'Preventive Measures': [], 'Long-Term Solutions': []}
    else:
        # Gemini not available: return a template if present
        tpl = CARE_PLAN_TEMPLATES.get(clean_disease_name)
        if tpl:
            care_plan_structured = tpl
            care_plan = "\n\n".join([
                "### Immediate Actions\n- " + "\n- ".join(tpl['Immediate Actions']),
                "### Preventive Measures\n- " + "\n- ".join(tpl['Preventive Measures']),
                "### Long-Term Solutions\n- " + "\n- ".join(tpl['Long-Term Solutions'])
            ])
            try:
                CARE_PLAN_CACHE[cache_key] = (care_plan, care_plan_structured)
            except Exception:
                pass
            return care_plan, care_plan_structured
        return "Gemini is not configured. No care plan available.", {'Immediate Actions': [], 'Preventive Measures': [], 'Long-Term Solutions': []}


def parse_care_plan_into_sections(text):
    sections = {
        'Immediate Actions': [],
        'Preventive Measures': [],
        'Long-Term Solutions': []
    }

    current = None
    if not text:
        return sections

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if 'immediate action' in low:
            current = 'Immediate Actions'
            continue
        if 'preventive measure' in low:
            current = 'Preventive Measures'
            continue
        if 'long-term' in low or 'long term' in low:
            current = 'Long-Term Solutions'
            continue

        if current:
            m = re.match(r'^\s*([-*•\u2022]|\d+\.)\s*(.+)', line)
            if m:
                item = m.group(2).strip()
                sections[current].append(item)
                continue

            cleaned = re.sub(r'^[^A-Za-z0-9]+', '', line).strip()
            if len(cleaned) > 5:
                sections[current].append(cleaned)
                continue

    return sections


# ----------------- New endpoint: on-demand recommendation via Gemini -----------------
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json() or {}
    disease = data.get('disease')
    risk = data.get('risk', 'Not available')

    if not disease:
        return jsonify({'error': 'disease is required'}), 400

    try:
        care_plan, care_plan_structured = generate_care_plan(disease, risk)
        return jsonify({'care_plan': care_plan, 'care_plan_structured': care_plan_structured})
    except Exception as e:
        print(f"Error in /recommend: {e}")
        return jsonify({'error': 'failed to generate care plan'}), 500


# ----------------- Chat message endpoint (uses chatbot.py) -----------------
@app.route('/chat_message', methods=['POST'])
def chat_message():
    """Accepts JSON { message, session_id } and returns JSON { response, session_id }.
    Uses the chatbot.initialize_gemini and generate_chat_response functions in src/chatbot.py.
    """
    global chat_model, CHAT_SESSIONS
    data = request.get_json() or {}
    message = data.get('message', '').strip()
    session_id = data.get('session_id') or f"sess_{int(time.time()*1000)}"

    if not message:
        return jsonify({'error': 'message is required'}), 400

    # Lazy initialize the Gemini chat model using the existing initialize_gemini() in chatbot.py
    try:
        if chat_model is None:
            chat_model = initialize_gemini()
            # If chatbot.initialize_gemini didn't return a model but app-level Gemini is initialized,
            # fall back to the shared model_gemini used for care plans (keeps chatbot.py unchanged).
            if chat_model is None and model_gemini is not None:
                chat_model = model_gemini
            # If still None, try to initialize app-level Gemini and use that model
            if chat_model is None and ensure_gemini_initialized():
                chat_model = model_gemini
    except Exception as e:
        print(f"Error initializing chat model: {e}")
        chat_model = None

    # Build a lightweight chat history string (last N turns) to send to the model
    history_entries = CHAT_SESSIONS.get(session_id, [])
    # Format history as simple alternating lines (user/bot)
    history_text = []
    for turn in history_entries[-10:]:
        u = turn.get('user', '')
        b = turn.get('bot', '')
        if u:
            history_text.append(f"User: {u}")
        if b:
            history_text.append(f"Bot: {b}")
    history_str = "\n".join(history_text)

    # Generate response (chatbot.generate_chat_response expects model, message, chat_history)
    resp_text = None
    # Prefer to use the app-level Gemini model so we can control language (English/Tamil/Tanglish)
    try:
        # Determine preferred language from the user's message
        def detect_language_hint(s):
            # If contains Tamil unicode range, prefer Tamil
            if re.search(r"[\u0B80-\u0BFF]", s):
                return 'Tamil'
            # common romanized Tamil tokens (simple heuristic)
            tanglish_tokens = ['vanakkam', 'vankam', 'vannakam', 'nalla', 'enna', 'ungal', 'ungalukku', 'nan', 'oru']
            low = s.lower()
            for t in tanglish_tokens:
                if t in low:
                    return 'Tanglish'
            # default to English
            return 'English'

        lang_hint = detect_language_hint(message)

        # If we have a shared model_gemini available, use it directly with a controlled prompt
        if model_gemini is not None:
            prompt = f"""
You are a friendly, practical agricultural assistant for smallholder farmers. Keep answers concise (1-5 short sentences).
Respond in the requested style/language: {lang_hint}.
If {lang_hint} is 'Tamil', reply in Tamil (Unicode). If 'Tanglish', reply using romanized Tamil (Latin letters) in a friendly conversational tone. If 'English', reply in English.

Conversation history:
{history_str}

User: {message}

Assistant:
"""
            try:
                resp = model_gemini.generate_content(prompt)
                resp_text = getattr(resp, 'text', None) or str(resp)
                print(f"chat_message: generated with model_gemini (lang_hint={lang_hint})")
            except Exception as e:
                print(f"Error generating chat via model_gemini: {e}")
                resp_text = None

        # If model_gemini wasn't available or failed, fall back to the existing chatbot helper
        if not resp_text:
            try:
                resp_text = generate_chat_response(chat_model, message, history_str)
            except Exception as e:
                print(f"Error during chat generation fallback: {e}")
                resp_text = "மன்னிக்கவும், ஒரு பிழை ஏற்பட்டது. தயவுசெய்து மீண்டும் முயற்சிக்கவும்."
    except Exception as e:
        print(f"Unexpected error in chat_message handler: {e}")
        resp_text = "மன்னிக்கவும், ஒரு பிழை ஏற்பட்டது. தயவுசெய்து மீண்டும் முயற்சிக்கவும்."

    # Persist session (append latest turn)
    try:
        history_entries.append({'user': message, 'bot': resp_text})
        CHAT_SESSIONS[session_id] = history_entries
    except Exception:
        pass

    return jsonify({'response': resp_text, 'session_id': session_id})


if __name__ == "__main__":
    # Pre-warm Gemini initialization in a background thread to reduce first-request latency.
    def _prewarm():
        try:
            print("Pre-warming Gemini in background thread...")
            ensure_gemini_initialized()
        except Exception as e:
            print(f"Pre-warm Gemini failed: {e}")

    t = threading.Thread(target=_prewarm, daemon=True)
    t.start()

    # Run Flask app. Consider setting debug=False in production to avoid double imports.
    app.run(host='0.0.0.0', port=5000, debug=True)

