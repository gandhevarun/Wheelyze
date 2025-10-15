from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from final_combine import predict_tyre_custom_ensemble

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')  # Must match your HTML file name

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        predictions = predict_tyre_custom_ensemble(file_path)
        result = predictions.get('Ensemble_prediction', None)
        return jsonify({'prediction': result})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
