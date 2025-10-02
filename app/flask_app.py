from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re

app = Flask(__name__)
model = joblib.load("models/udemy_subscribers_pipeline.joblib")

def simple_clean_title(t):
    t = str(t).lower().strip()
    t = re.sub(r'[^\w\s]','',t)
    return t

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error":"No input data"}), 400

    data['title'] = simple_clean_title(data.get('title',''))
    input_df = pd.DataFrame([data])
    pred = model.predict(input_df)[0]
    return jsonify({"predicted_num_subscribers": int(round(pred))})

if __name__ == "__main__":
    app.run(debug=True)
