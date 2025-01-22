# machine-downtime-predictive-api
A FastAPI-based RESTful API for predicting machine downtime in manufacturing operations.

## Description

This project provides a RESTful API for predicting machine downtime using a Gradient Boosting Classifier. The API allows users to upload manufacturing data, train the model, and make predictions on new data.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Avdeep-Singh/machine-downtime-predictive-api
2. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
## Usage
1. **Running the API**
    ```bash
    uvicorn app:app --reload
The API will be accessible at http://localhost:8000

## API Endpoints
1. **Upload Data (POST /upload):**
Uploads a CSV file containing manufacturing data.
- **Request:**
    ```bash
    curl -X POST -F file=@/path/to/your/file.csv http://localhost:8000/upload
    ```
    (Replace /path/to/your/file.csv with the actual path to your CSV file).
- **Output**
   - Response (Success):
     ```json
     {"message": "Data uploaded successfully."}
   - Response (Error - Invalid File Type):**
     ```json
     {"detail": "Invalid file type. Only CSV files are allowed."}
2. **Train Model (POST /train):**
Trains the Gradient Boosting model on the uploaded data.
- **Request:**
    ```bash
    curl -X POST http://localhost:8000/train
- **Output**
   - Response (Success):
     ```json
     {"accuracy": 0.95, "f1_score": 0.92}
   - Response (Error - No Data Uploaded):
     ```json
     {"detail": "No data uploaded yet. Please upload a CSV file first."}
3. **Predict Downtime (POST /predict):**
Makes a prediction on new data.
- **Request:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"Machine_ID": "Makino-L1-Unit1-2013", "Assembly_Line_No": "Shopfloor-L1", "Hydraulic_Pressure(bar)": 71.04, "Coolant_Pressure(bar)": 6.933724915, "Air_System_Pressure(bar)": 6.284964506, "Coolant_Temperature": 25.6, "Hydraulic_Oil_Temperature(?C)": 46.0, "Spindle_Bearing_Temperature(?C)": 33.4, "Spindle_Vibration(?m)": 1.291, "Tool_Vibration(?m)": 26.492, "Spindle_Speed(RPM)": 25892.0, "Voltage(volts)": 335.0, "Torque(Nm)": 24.05532601, "Cutting(kN)": 3.58}' http://localhost:8000/predict
    ```
    (Replace placeholders with actual feature values. Include all features used in training).
- **Output**
   - Response (Success):
     ```json
     {"Downtime": "No", "Confidence": 0.85} # Example values
     ```
     (Downtime 'No' means no fault in machine and 'Yes' means fault in machine)
     (Assuming 0 maps to "Yes" which means Machine_Failure)
   - Response (Error - Model Not Trained):
     ```json
     {"detail": "Model not trained. Please train the model first."}

## Limitations and Assumptions

1. **Edge Case Handling**:  
   The current implementation does not handle edge cases such as:
   - Empty or malformed datasets during upload.  
   - Missing values in the prediction input.  
   These were omitted to focus on the core functionality. In production, robust input validation and error handling would be essential.

2. **Unseen Categories in Predictions**:  
   The model uses `LabelEncoder` for encoding categorical features. The `/predict` endpoint assumes all categorical input values are consistent with the training data. Predictions with unseen categories may raise errors. This could be mitigated by:
   - Adding logic to handle unknown categories dynamically.  
   - Re-training the encoder with new inputs.  

3. **Static Preprocessing**:  
   The preprocessing steps (e.g., dropping null values and the `Date` column) are hard-coded to suit the provided sample dataset. For generalization, these steps would need to be dynamic, based on dataset characteristics.  

4. **Dataset-Specific Assumptions**:  
   The API is designed to work with the sample dataset provided (`Sample_Data_from_kaggle.csv`). Additional preprocessing or modifications may be required to handle datasets with different formats or distributions.

## Data
A sample dataset (Sample_Data_from_kaggle.csv) is included in the root directory.
The target variable is Downtime, where 0 represents machine failure ("Yes") and 1 represents no failure ("No").
The dataset contains the following features:
- Machine_ID
- Assembly_Line_No
- Hydraulic_Pressure(bar)
- Coolant_Pressure(bar)
- Air_System_Pressure(bar)
- Coolant_Temperature
- Hydraulic_Oil_Temperature(?C)
- Spindle_Bearing_Temperature(?C)
- Spindle_Vibration(?m)
- Tool_Vibration(?m)
- Spindle_Speed(RPM)
- Voltage(volts)
- Torque(Nm)
- Cutting(kN)
- Downtime

## Model
A Gradient Boosting Classifier from scikit-learn is used for prediction.
- Model parameters: 
    ***random_state = 42***
- Preprocessing and Feature Engineering:
    - Data Cleaning: The dataset is assumed to be pre-cleaned, meaning that it contains no duplicates, and obvious data inconsistencies have been addressed. Missing values are dropped and the Data column is also dropped.
    - Categorical Encoding: Label Encoding is used to convert categorical features to numerical representations.

## Error Handling
The API includes error handling for invalid file uploads, missing data, and untrained models and other basic error handling is included which will help the user to understand the error.

## Testing
Automated tests are included in test_app.py. To run the tests:
```bash
pytest
```
