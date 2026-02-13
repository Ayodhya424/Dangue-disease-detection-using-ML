import os
import io
import re
import fitz  # PyMuPDF
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import joblib
import numpy as np

# Optional Tesseract import
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

app = Flask(__name__)
model = joblib.load('rf_dengue_model.pkl')

# Field metadata with display names and ranges
FIELD_METADATA = {
    'Gender': {'type': 'select', 'options': ['Male', 'Female'], 'range': 'Male/Female'},
    'Age': {'type': 'number', 'range': '1-120 years', 'placeholder': 'e.g., 25'},
    'Hemoglobin(g/dl)': {'type': 'number', 'range': '7-20 g/dl', 'placeholder': 'e.g., 12.5'},
    'Neutrophils(%)': {'type': 'number', 'range': '0-100%', 'placeholder': 'e.g., 60'},
    'Lymphocytes(%)': {'type': 'number', 'range': '0-100%', 'placeholder': 'e.g., 30'},
    'Monocytes(%)': {'type': 'number', 'range': '0-100%', 'placeholder': 'e.g., 5'},
    'Eosinophils(%)': {'type': 'number', 'range': '0-10%', 'placeholder': 'e.g., 2'},
    'RBC': {'type': 'number', 'range': '3.5-6.0 (Million/µL)', 'placeholder': 'e.g., 4.5'},
    'HCT(%)': {'type': 'number', 'range': '30-55%', 'placeholder': 'e.g., 42'},
    'MCV(fl)': {'type': 'number', 'range': '80-100 fL', 'placeholder': 'e.g., 90'},
    'MCH(pg)': {'type': 'number', 'range': '27-33 pg', 'placeholder': 'e.g., 30'},
    'MCHC(g/dl)': {'type': 'number', 'range': '32-36 g/dl', 'placeholder': 'e.g., 34'},
    'RDW-CV(%)': {'type': 'number', 'range': '11-15%', 'placeholder': 'e.g., 13'},
    'Total Platelet Count(/cumm)': {'type': 'number', 'range': '150k-450k', 'placeholder': 'e.g., 250000'},
    'MPV(fl)': {'type': 'number', 'range': '7-11 fL', 'placeholder': 'e.g., 9'},
    'PDW(%)': {'type': 'number', 'range': '10-18%', 'placeholder': 'e.g., 15'},
    'PCT(%)': {'type': 'number', 'range': '0.15-0.40%', 'placeholder': 'e.g., 0.25'},
    'Total WBC count(/cumm)': {'type': 'number', 'range': '4k-11k', 'placeholder': 'e.g., 7000'},
    'Fever': {'type': 'select', 'options': ['Yes', 'No'], 'range': 'Yes/No'},
    'Severe_Body_Pain': {'type': 'select', 'options': ['Yes', 'No'], 'range': 'Yes/No'},
    'Headache': {'type': 'select', 'options': ['Yes', 'No'], 'range': 'Yes/No'},
    'Rash': {'type': 'select', 'options': ['Yes', 'No'], 'range': 'Yes/No'},
    'Bleeding_Signs': {'type': 'select', 'options': ['Yes', 'No'], 'range': 'Yes/No'},
    'Vomiting': {'type': 'select', 'options': ['Yes', 'No'], 'range': 'Yes/No'}
}

FEATURES = list(FIELD_METADATA.keys())

def extract_text_from_file(file):
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "".join([page.get_text() for page in doc])
    elif filename.endswith(('.png', '.jpg', '.jpeg')):
        if not TESSERACT_AVAILABLE:
            # Return empty string instead of raising error - fallback to manual entry
            return None
        try:
            image = Image.open(file)
            return pytesseract.image_to_string(image)
        except Exception:
            return None
    return ""

def parse_medical_text(text):
    """Extract medical values from text using regex."""
    if text is None or not text.strip():
        return {}
    
    extracted_data = {}
    for feature in FEATURES:
        pattern = re.compile(rf"{re.escape(feature)}[:\s]*(\d+\.?\d*)", re.IGNORECASE)
        match = pattern.search(text)
        extracted_data[feature] = float(match.group(1)) if match else None
    return extracted_data

def convert_form_data(form_data):
    """Convert form data to model input format."""
    data = {}
    for feature in FEATURES:
        value = form_data.get(feature, '')
        if not value:
            return None, f"Missing value for {feature}"
        
        # Convert Yes/No to 1/0, Male/Female to 1/0
        if value.lower() in ['yes', 'male']:
            data[feature] = 1
        elif value.lower() in ['no', 'female']:
            data[feature] = 0
        else:
            try:
                data[feature] = float(value)
            except ValueError:
                return None, f"Invalid value for {feature}: {value}"
    
    return data, None

def make_prediction(data):
    """Make dengue prediction using the model."""
    if data is None:
        return None, None
    
    feature_array = np.array([data[feature] for feature in FEATURES]).reshape(1, -1)
    
    try:
        prediction = model.predict(feature_array)[0]
        probability = model.predict_proba(feature_array)[0]
        
        result = "🔴 DENGUE POSITIVE" if prediction == 1 else "🟢 DENGUE NEGATIVE"
        confidence = max(probability) * 100
        
        return result, round(confidence, 2)
    except Exception as e:
        return None, None

@app.route('/')
def home():
    return render_template('index.html', features=FEATURES, field_metadata=FIELD_METADATA)

@app.route('/download-template')
def download_template():
    """Download CSV template for easy data entry."""
    df = pd.DataFrame(columns=FEATURES)
    # Add example row
    df.loc[0] = ['Male', 30, 13.5, 65, 30, 3, 2, 4.5, 42, 90, 30, 34, 13, 250000, 9, 15, 0.25, 7000, 'Yes', 'Yes', 'No', 'No', 'No', 'No']
    
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='dengue_report_template.csv'
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('file')
        
        # If file is uploaded, extract data and show form for verification
        if file and file.filename:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                if len(df) == 0:
                    return render_template('index.html', 
                                         features=FEATURES, 
                                         field_metadata=FIELD_METADATA,
                                         error="CSV file is empty. Please add patient data.")
                
                extracted_list = df.to_dict('records')
                return render_template('index.html', 
                                     features=FEATURES, 
                                     field_metadata=FIELD_METADATA,
                                     msg=f"Loaded {len(extracted_list)} record(s) from CSV. Using first record.",
                                     extracted=extracted_list[0])
            else:
                # Handle Image or PDF
                raw_text = extract_text_from_file(file)
                
                if raw_text is None:
                    # OCR not available
                    return render_template('index.html',
                                         features=FEATURES,
                                         field_metadata=FIELD_METADATA,
                                         error="⚠️ Image text extraction not available (Tesseract not installed). Please use CSV format or fill the form manually. <a href='/download-template' class='alert-link'>Download CSV template</a>")
                
                if not raw_text.strip():
                    # Extraction failed
                    return render_template('index.html',
                                         features=FEATURES,
                                         field_metadata=FIELD_METADATA,
                                         error="Could not extract text from file. The image may be unclear or in an unsupported format. Please try: 1) Upload PDF instead, 2) Upload CSV file, or 3) Fill form manually. <a href='/download-template' class='alert-link'>Download CSV template</a>")
                
                extracted_data = parse_medical_text(raw_text)
                clean_extracted = {k: v for k, v in extracted_data.items() if v is not None}
                
                if not clean_extracted:
                    return render_template('index.html',
                                         features=FEATURES,
                                         field_metadata=FIELD_METADATA,
                                         error="No medical data found in the file. Please try: 1) A clearer image, 2) Upload as PDF, 3) Upload CSV file, or 4) Fill form manually. <a href='/download-template' class='alert-link'>Download CSV template</a>")
                
                return render_template('index.html', 
                                     extracted=clean_extracted, 
                                     features=FEATURES, 
                                     field_metadata=FIELD_METADATA,
                                     msg="✅ Data extracted from document. Please verify and correct values before predicting.")
        
        # If no file, process form data
        data, error = convert_form_data(request.form)
        if error:
            return render_template('index.html',
                                 features=FEATURES,
                                 field_metadata=FIELD_METADATA,
                                 error=error)
        
        result, confidence = make_prediction(data)
        
        if result is None:
            return render_template('index.html',
                                 features=FEATURES,
                                 field_metadata=FIELD_METADATA,
                                 error="Error making prediction. Please check your data.")
        
        return render_template('index.html',
                             features=FEATURES,
                             field_metadata=FIELD_METADATA,
                             result=result,
                             confidence=confidence)

    except Exception as e:
        return render_template('index.html',
                             features=FEATURES,
                             field_metadata=FIELD_METADATA,
                             error=f"Processing Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)