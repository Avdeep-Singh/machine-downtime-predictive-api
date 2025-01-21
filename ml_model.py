import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

class DowntimePredictor:
    def __init__(self, model_path="model/trained_model.pkl"):
        self.model_path = model_path
        try:
            loaded_data = joblib.load(self.model_path)
            self.model = loaded_data["model"]
            self.encoders = loaded_data["encoders"]
        except FileNotFoundError:
            self.model = None

    def load_data(self, file_to_upload, filepath="data/uploaded_data.csv"):
        try:
            df = pd.read_csv(file_to_upload.file)
            df.to_csv(filepath, index=False) # Save to data directory
            return "Data uploaded successfully."
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def train(self, df, target_column="Downtime"):
        try:
            df = df.drop(['Date'], axis=1) # droping date column
            df = df.dropna() # droping null values
            
            # encoding object values to int64
            self.encoders = {}
            for column in df.columns:
                if df[column].dtype == type(object):
                    le = LabelEncoder()
                    df[column] = le.fit_transform(df[column])
                    if column != 'Downtime':
                        self.encoders[column] = le

            # splitting data into x and y
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            self.model = GradientBoostingClassifier(random_state=42)
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            joblib.dump({"model": self.model, "encoders": self.encoders}, self.model_path) # saving trained model and encoder 

            return {"accuracy": accuracy, "f1_score": f1}
        except Exception as e:
            raise Exception(f"Error training model: {e}")

    def predict(self, input_data):
        if self.model is None:
            raise Exception("Model not trained. Please train the model first.")
        try:
            input_df = pd.DataFrame([input_data])
            for column, le in self.encoders.items(): # encoding input object values 
                input_df[column] = le.transform(input_df[column])
            prediction = self.model.predict(input_df)[0]
            confidence = self.model.predict_proba(input_df)[0][prediction]
            prediction_label = "Yes" if prediction == 0 else "No"  # Assuming 0 maps to "Yes" which means Machine_Failure
            return {"Downtime": prediction_label, "Confidence": confidence}
        except Exception as e:
            raise Exception(f"Error making prediction: {e}")