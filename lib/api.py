import uvicorn
import joblib
from .encoding import GeneEncoder
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("/home/marcello/learn/python/AntibioticResistance/model/resistance_model.keras")

# Define input schema
class InputData(BaseModel):
    feature: str  # single sequence input

# Initialize FastAPI
app = FastAPI(title="Keras Model API", description="Serve predictions via FastAPI")

@app.post("/predict")
def predict(data: InputData):
    sequence = data.feature  # extract the sequence string

    # TODO: Preprocess the sequence into numeric features
    # Example: turn chars into ASCII codes (placeholder)
    max_len = 200  # ⚠️ must match training
    encoder = GeneEncoder()
    X_new = encoder.encode_protein_batch([sequence], max_len=max_len)

    # Predict
    pred = model.predict(X_new)
    label_encoder = joblib.load("/home/marcello/learn/python/AntibioticResistance/model/label_encoder.pkl")
    predicted_class = label_encoder.inverse_transform(pred.argmax(axis=1))

    class_str = list(predicted_class)

    print(class_str[0])
    return {"prediction": class_str[0]}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
