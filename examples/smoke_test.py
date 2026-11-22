"""Exercise the downtime API with synthetic data in an isolated directory."""

import csv
import io
from contextlib import chdir
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient


def synthetic_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "pressure", "temperature", "vibration", "Downtime"])
    for row in range(80):
        writer.writerow(
            ["2025-01-01", 50 + row, 25 + row % 8, (row % 5) / 10, int(row >= 40)]
        )
    return output.getvalue()


def main():
    with TemporaryDirectory(prefix="downtime-smoke-") as scratch, chdir(scratch):
        Path("data").mkdir()
        Path("model").mkdir()

        # Import here so startup sees only the empty temporary model directory.
        from app import app

        with TestClient(app) as client:
            upload = client.post(
                "/upload", files={"file": ("synthetic.csv", synthetic_csv(), "text/csv")}
            )
            upload.raise_for_status()
            assert upload.json()["message"] == "Data uploaded successfully."

            training = client.post("/train")
            training.raise_for_status()
            assert {"accuracy", "f1_score"} <= training.json().keys()

            prediction = client.post(
                "/predict", json={"pressure": 55, "temperature": 28, "vibration": 0.2}
            )
            prediction.raise_for_status()
            result = prediction.json()
            assert result["Downtime"] in {"Yes", "No"}
            assert 0 <= result["Confidence"] <= 1

        assert Path("model/trained_model.pkl").is_file()
        print("Synthetic upload, train, and predict checks passed.")


if __name__ == "__main__":
    main()
