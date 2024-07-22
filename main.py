import pickle
import numpy as np

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import Union

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.preprocessor import casefolding, text_normalize, remove_stopwords, stemming, lemmatization

app = FastAPI()

class Feedback(BaseModel):
    email: Union[str, None] = None
    text: str

# Function to preprocess text
def preprocess_text(text):
    # text = re.sub(r'\bsampah\b', 'sampahnya', text)
    text = casefolding(text)
    text = text_normalize(text)
    text = remove_stopwords(text)
    text = stemming(text)
    text = lemmatization(text)
    return text

@app.get("/", description="This is our first route.")
async def root():
    return {"message": "Hello we are from Group 2"}

@app.post("/api/sentiment", description="This is sentiment analysis route")
async def post(feedback: Feedback):
    text = feedback.text

    text_processed = preprocess_text(text)

    # Load tokenizer's configuration
    with open('../be-sentiment-pk/models/asli_fix/tokenizer_config_w2v_bilstm_fix.pkl', 'rb') as f:
        tokenizer_config = pickle.load(f)

    # Recreate tokenizer
    tokenizer = Tokenizer(**tokenizer_config)

    # Load tokenizer's word index
    with open('../be-sentiment-pk/models/asli_fix/tokenizer_word_index_w2v_bilstm_fix.pkl', 'rb') as f:
        tokenizer.word_index = pickle.load(f)

    # new data to predict
    new_texts = [text_processed]

    # Tokenize new data
    sequences = tokenizer.texts_to_sequences(new_texts)

    # Padding sequences
    max_length = 100  # model's max_length
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    # Load the saved model
    model = load_model('../be-sentiment-pk/models/asli_fix/model_w2v_bilstm_fix.keras')

    # Perform prediction
    predictions = model.predict(padded_sequences)

    # Convert predictions to a list for JSON serialization
    predictions_list = predictions.tolist()
    result_predictions = {
        "negative": predictions_list[0][0],
        "neutral": predictions_list[0][1],
        "positive": predictions_list[0][2]
    }

    return {
        "message": "Model accessed successfully",
        "text": text,
        "predictions": result_predictions
    }

# if __name__ == "__main__":
#   uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)