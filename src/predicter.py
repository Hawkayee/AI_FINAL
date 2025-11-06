# predicter.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
import torch.serialization
import google.generativeai as genai
from utils.secrets import load_api_key
import sys
from ultralytics import YOLO

# ------------------- Gemini API Setup -------------------
API_KEY = load_api_key('GEMINI_API_KEY')
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        preferred_model = load_api_key('GEMINI_MODEL')
        if preferred_model:
            try:
                model_gemini = genai.GenerativeModel(preferred_model)
            except Exception as e:
                print(f"Could not initialize GEMINI_MODEL in predicter='{preferred_model}': {e}")
                model_gemini = None
        else:
            try:
                available_models = genai.list_models() or []
                gen_models = [m for m in available_models if getattr(m, 'supported_generation_methods', []) and 'generateContent' in m.supported_generation_methods]
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
                        print(f"Predicter initialized Gemini model: {chosen}")
                    except Exception as e:
                        print(f"Predicter failed to initialize auto-chosen model '{chosen}': {e}")
                        model_gemini = None
                else:
                    print("Predicter: No suitable generateContent models found. Gemini disabled.")
                    model_gemini = None
            except Exception as e:
                print(f"Predicter error while listing models: {e}")
                model_gemini = None
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        model_gemini = None
else:
    print("Warning: GEMINI_API_KEY not found in env/.env/secrets.json. Gemini features disabled in predicter.")
    model_gemini = None

# ------------------- Paths -------------------
model_path = "model/plant_disease_model.pth"
classes_path = "model/classes.json"
yolo_model_path = "yolov8n-oiv7.pt"

# ------------------- Load Classes -------------------
with open(classes_path, "r") as f:
    classes = json.load(f)

# ------------------- Model Architecture -------------------
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
        
# ------------------- Load Models -------------------
torch.serialization.add_safe_globals([CNN_NeuralNet])
disease_model = torch.load(model_path, map_location="cpu", weights_only=False)
disease_model.eval()
yolo_model = YOLO(yolo_model_path)

PLANT_CLASSES = {'Flower', 'Houseplant', 'Plant', 'Tree', 'Leaf'}
CONFIDENCE_THRESHOLD_YOLO = 0.20

transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# ------------------- Prediction + Gemini -------------------
def predict_and_generate_plan(image_path):
    if not os.path.exists(image_path):
        return {"error": f"Image '{image_path}' not found!"}

    image = Image.open(image_path).convert("RGB")

    # Step 1: Use YOLOv8 to detect if the image is a plant
    yolo_results = yolo_model.predict(source=image, verbose=False)
    is_plant_detected = any(
        yolo_model.names[int(box.cls)] in PLANT_CLASSES and box.conf > CONFIDENCE_THRESHOLD_YOLO
        for result in yolo_results for box in result.boxes
    )

    if not is_plant_detected:
        return {"error": "The uploaded image does not appear to be a plant."}

    # Step 2: If it is a plant, proceed with disease classification
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = disease_model(img_t)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    disease = classes[predicted.item()]
    confidence_score = round(confidence.item() * 100, 2)
    clean_disease_name = disease.replace('_', ' ').replace('___', ' ').title()

    # CORRECTED: A much stricter prompt for clean, predictable formatting
    prompt = f"""
    Generate a farmer-friendly care plan for a plant with "{clean_disease_name}".

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

    try:
        gemini_response = model_gemini.generate_content(prompt)
        care_plan = gemini_response.text if gemini_response else "No care plan available."
    except Exception as e:
        care_plan = f"Error fetching care plan: {e}"

    return {
        "disease": clean_disease_name,
        "confidence": confidence_score,
        "care_plan": care_plan
    }

# ------------------- Run Dynamically -------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predicter.py <image_path>") # Corrected usage message
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict_and_generate_plan(image_path)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Detected Disease: {result['disease']}")
        print(f"Confidence: {result['confidence']}%")
        print("\nCare Plan:\n")
        print(result["care_plan"])