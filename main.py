import pickle
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

class Feedback(BaseModel):
    email: Union[str, None] = None
    text: str

@app.get("/", description="This is our first route.")
async def root():
    return {"message": "Hello we are from Group 2"}

@app.post("/api/sentiment", description="This is sentiment analysis route")
async def post(text: str):
    # Load tokenizer's configuration
    with open('./model/tokenizer_config.pkl', 'rb') as f:
        tokenizer_config = pickle.load(f)

    # Recreate tokenizer
    tokenizer = Tokenizer(**tokenizer_config)

    # Load tokenizer's word index
    with open('./model/tokenizer_word_index.pkl', 'rb') as f:
        tokenizer.word_index = pickle.load(f)

    # new data to predict
    new_texts = [text]

    # Tokenize new data
    sequences = tokenizer.texts_to_sequences(new_texts)

    # Padding sequences
    max_length = 100  # model's max_length
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    # Load the saved model
    model = load_model('./model/model_lstm.keras')

    # Perform prediction
    predictions = model.predict(padded_sequences)

    # Convert predictions to a list for JSON serialization
    predictions_list = predictions.tolist()
    result_predictions = {
        "score_negative": predictions_list[0][0],
        "score_neutral": predictions_list[0][1],
        "score_positive": predictions_list[0][2]
    }

    return {
        "message": "Model accessed successfully",
        "text": text,
        "predictions": result_predictions
    }