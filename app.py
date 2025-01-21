from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
from ml_model import DowntimePredictor

app = FastAPI()
predictor = DowntimePredictor() # Initialize the predictor
ALLOWED_EXTENSIONS = {"csv"}  # Set of allowed file extensions

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/upload") # endpoint for uploading CSV file only
async def upload_data(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are allowed.")
    try:
        message = predictor.load_data(file_to_upload = file)
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading data: {e}")

@app.post("/train") # endpoint for training
async def train_model():
    try:
        import os
        if not os.path.exists("data/uploaded_data.csv"):
            raise HTTPException(status_code=400, detail="No data uploaded yet. Please upload a CSV file first.")
        df = pd.read_csv("data/uploaded_data.csv")
        metrics = predictor.train(df)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {e}")


@app.post("/predict") # endpoint for predictions 
async def predict_downtime(input_data: dict):
    try:
        prediction = predictor.predict(input_data)
        return JSONResponse(content=prediction) # Return JSON response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}") 


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)