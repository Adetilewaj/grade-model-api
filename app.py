import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load('grade_model.pkl')
model_features = joblib.load('model_features.pkl')  # list of feature names

@app.route('/')
def home():
    return "Grade model is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Build features array in correct order
    try:
        # Extract features from incoming JSON
        input_features = data['features']  # should be a dict: {"feature_name": value, ...}
        
        # Create feature vector in correct order
        features_vector = [input_features[feat] for feat in model_features]
    except KeyError as e:
        return jsonify({'error': f'Missing feature in input: {e}'}), 400

    features_array = np.array(features_vector).reshape(1, -1)
    prediction = model.predict(features_array)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
