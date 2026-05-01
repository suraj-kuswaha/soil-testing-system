from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import datetime
import random

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# =========================
# LOAD & TRAIN MODEL
# =========================

data = pd.read_csv("soil_data.csv")

X = data[['N','P','K','temperature','humidity','ph','moisture']]
y = data['label']

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

print("✅ ML Model Loaded")

# =========================
# ROUTE 1: SENSOR DATA INPUT
# =========================

@app.route('/sensor-data', methods=['POST'])
def sensor_data():

    data = request.get_json()

    try:
        N = data['N']
        P = data['P']
        K = data['K']
        temp = data['temperature']
        humidity = data['humidity']
        ph = data['ph']
        moisture = data['moisture']

        # ML Prediction using DataFrame to avoid feature name warnings
        input_df = pd.DataFrame([[N, P, K, temp, humidity, ph, moisture]], 
                                columns=['N','P','K','temperature','humidity','ph','moisture'])
        prediction = model.predict(input_df)[0]

        # Timestamp
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save to file (simple database)
        with open("data_log.csv", "a") as f:
            f.write(f"{time},{N},{P},{K},{temp},{humidity},{ph},{moisture},{prediction}\n")

        return jsonify({
            "status": "success",
            "crop": prediction
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# =========================
# ROUTE 2: GET LATEST DATA (for frontend)
# =========================

@app.route('/latest', methods=['GET'])
def latest():

    try:
        df = pd.read_csv("data_log.csv")

        last = df.iloc[-1]

        return jsonify({
            "time": last[0],
            "N": last[1],
            "P": last[2],
            "K": last[3],
            "temperature": last[4],
            "humidity": last[5],
            "ph": last[6],
            "moisture": last[7],
            "crop": last[8]
        })

    except:
        return jsonify({"message": "No data available yet"})

# =========================
# ROUTE 3: AI CROP SCANNER
# =========================

@app.route('/analyze-crop', methods=['POST'])
def analyze_crop():
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image uploaded"})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected image"})

    # MOCK AI LOGIC
    # In a real app, you would pass the image to a deep learning model (e.g., PyTorch/TensorFlow)
    # or an external API like Google Gemini Vision API.
    # Here we randomly select a plausible diagnosis to demonstrate the flow.
    mock_analyses = [
        {
            "crop": "Wheat",
            "diagnosis": "Healthy",
            "solution": "Your crop looks great! Maintain current watering schedule and soil nutrients."
        },
        {
            "crop": "Maize",
            "diagnosis": "Nitrogen Deficiency",
            "solution": "Leaves show yellowing. Apply a nitrogen-rich fertilizer like Urea (46-0-0) to the soil."
        },
        {
            "crop": "Rice",
            "diagnosis": "Leaf Blast Disease",
            "solution": "Fungal infection detected. Apply a suitable fungicide containing Tricyclazole or Isoprothiolane and improve water management."
        },
        {
            "crop": "Tomato",
            "diagnosis": "Early Blight",
            "solution": "Fungal disease found on lower leaves. Remove affected leaves and apply a copper-based fungicide to prevent spreading."
        }
    ]

    analysis_result = random.choice(mock_analyses)

    return jsonify({
        "status": "success",
        "data": analysis_result
    })

# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":
    app.run(debug=True)