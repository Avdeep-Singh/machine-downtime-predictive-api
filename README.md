# Machine downtime prediction API

A FastAPI demo for uploading a labeled CSV, training a scikit-learn gradient boosting classifier, and requesting downtime predictions.

## Setup

Use Python 3.12. From the repository root, create and activate a virtual environment:

```sh
python -m venv .venv
```

Use `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS or Linux. Then run:

```sh
python -m pip install -r requirements.txt
python -m examples.smoke_test
```

The smoke test creates synthetic data and exercises all three endpoints in process. It runs from a temporary working directory, trains a fresh model, and leaves the committed CSVs and model untouched. It checks the API contract without using its scores as evidence of real-world performance.

## API flow

| Step | Request | Response |
|---|---|---|
| Upload | `POST /upload` with multipart field `file` containing a CSV | A confirmation message |
| Train | `POST /train` with no request body | Holdout accuracy and F1 score |
| Predict | `POST /predict` with a JSON object containing the training features | `Downtime` and `Confidence` |

The smoke test uses three numeric feature columns, plus `Date` and the binary `Downtime` label. The training code drops `Date`. Send the same features in the same order at prediction time and omit both `Date` and `Downtime`. For instance:

```json
{
  "pressure": 55,
  "temperature": 28,
  "vibration": 0.2
}
```

Use both target classes when training. The prediction code maps label `0` to downtime `Yes` and label `1` to `No`. The returned confidence is a model probability. Calibration has not been established.

## Running the server

The app reads and writes paths relative to its working directory. For a fresh run, start it from a separate empty directory with empty `data/` and `model/` subdirectories. Keep the virtual environment active and point Uvicorn at the source checkout:

```sh
python -m uvicorn app:app --app-dir "/absolute/path/to/machine-downtime-predictive-api" --host 127.0.0.1
```

Open `http://127.0.0.1:8000/docs` to exercise the endpoints. Upload your CSV, train, then predict. The API writes the uploaded data to `data/uploaded_data.csv` and the trained model to `model/trained_model.pkl`. It loads a model from that path at startup if present. Use only model artifacts you created or trust.

## Data and limitations

The repository contains two existing CSVs and a serialized model. The original sample filename references Kaggle. The exact dataset listing, redistribution terms, and model training inputs are unverified. The synthetic walkthrough is independent of those artifacts.

The demo uses one local upload and one active model. The current preprocessing assumes a consistent schema and may reject missing fields or unseen categories. It is intended for local demonstration.
