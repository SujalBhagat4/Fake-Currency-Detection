from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import base64
from io import BytesIO
from PIL import Image
import json
from datetime import datetime
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model variable (load once at startup)
model = None

def load_detection_model():
    """Load the trained model - update the path to your model"""
    global model
    try:
        # Update this path to your actual model location
        model_path = 'model.keras'  # Place your model file here
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure your model file is in the correct location")
        return False

def preprocess_image(image_bgr):
    """Resize & normalize for the model"""
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, axis=0).astype(np.float32)

def predict_currency(image_bgr):
    """Return the probability of Fake currency (0–1)"""
    if model is None:
        raise Exception("Model not loaded")
    
    processed = preprocess_image(image_bgr)
    pred = model.predict(processed, verbose=0)
    score = float(pred[0][0])
    return score

def currency_prediction_majority(image_path, threshold=0.4):
    """Rotate image and predict with majority vote"""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    rotations = {
        "0°": image_bgr,
        "90°": cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE),
        "180°": cv2.rotate(image_bgr, cv2.ROTATE_180),
        "270°": cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }

    rotation_scores = {}
    rotation_labels = {}

    for angle, img in rotations.items():
        score = predict_currency(img)
        label = "Fake Currency" if score >= threshold else "Real Currency"
        rotation_scores[angle] = score
        rotation_labels[angle] = label

    # Majority vote
    fake_votes = sum(1 for lbl in rotation_labels.values() if lbl == "Fake Currency")
    real_votes = len(rotation_labels) - fake_votes
    final_label = "Fake Currency" if fake_votes >= real_votes else "Real Currency"

    return final_label, rotation_scores, rotation_labels, fake_votes, real_votes

def generate_pdf_report(image_path, final_label, scores, labels, fake_votes, real_votes):
    """Generate PDF report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"currency_report_{timestamp}.pdf"
    report_path = os.path.join(REPORTS_FOLDER, report_filename)
    
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Currency Authentication Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Timestamp
    timestamp_p = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
    story.append(timestamp_p)
    story.append(Spacer(1, 12))
    
    # Final Result
    result_style = styles['Heading2']
    if final_label == "Fake Currency":
        result_text = f'<font color="red">RESULT: {final_label}</font>'
    else:
        result_text = f'<font color="green">RESULT: {final_label}</font>'
    
    result = Paragraph(result_text, result_style)
    story.append(result)
    story.append(Spacer(1, 12))
    
    # Vote summary
    vote_summary = Paragraph(f"Vote Summary: Fake votes: {fake_votes}, Real votes: {real_votes}", styles['Normal'])
    story.append(vote_summary)
    story.append(Spacer(1, 12))
    
    # Rotation scores
    scores_text = "Detailed Scores by Rotation:<br/><br/>"
    for angle, score in scores.items():
        label = labels[angle]
        scores_text += f"{angle}: Score {score:.3f} → {label}<br/>"
    
    scores_p = Paragraph(scores_text, styles['Normal'])
    story.append(scores_p)
    story.append(Spacer(1, 12))
    
    # Add image if exists
    try:
        img = ReportImage(image_path, width=3*inch, height=3*inch)
        story.append(img)
    except:
        pass
    
    doc.build(story)
    return report_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Convert image to base64 for frontend display
        with open(filepath, 'rb') as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Make prediction
        final_label, scores, labels, fake_votes, real_votes = currency_prediction_majority(
            filepath, threshold=0.4
        )
        
        # Generate PDF report
        report_filename = generate_pdf_report(
            filepath, final_label, scores, labels, fake_votes, real_votes
        )
        
        # Calculate confidence score (average of all rotation scores)
        avg_score = np.mean(list(scores.values()))
        confidence = max(avg_score, 1 - avg_score) * 100  # Convert to confidence percentage
        
        result = {
            'final_label': final_label,
            'confidence': round(confidence, 1),
            'fake_votes': fake_votes,
            'real_votes': real_votes,
            'rotation_scores': scores,
            'rotation_labels': labels,
            'image_base64': img_base64,
            'report_filename': report_filename,
            'average_score': round(avg_score, 3)
        }
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_report/<filename>')
def download_report(filename):
    try:
        report_path = os.path.join(REPORTS_FOLDER, filename)
        return send_file(report_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': 'Report not found'}), 404

if __name__ == '__main__':
    print("Starting Currency Detection Flask App...")
    
    # Try to load the model
    if not load_detection_model():
        print("\n" + "="*50)
        print("WARNING: Model could not be loaded!")
        print("Please place your 'Fake-currency.keras' file in the same directory as app.py")
        print("The app will start but predictions will not work until the model is loaded.")
        print("="*50 + "\n")
    
    print("Server starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)