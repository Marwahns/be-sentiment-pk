import pickle
import numpy as np
import re
import pandas as pd

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import Union

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

class Feedback(BaseModel):
    email: Union[str, None] = None
    text: str

key_norm = pd.read_csv('../be-sentiment-pk/corpus/colloquial-indonesian-lexicon.csv')

def casefolding(string):
    string = string.lower()
    string = re.sub(r'https?://\S+|www\.\S+', '', string) # remove URLs
    string = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", '', string) # remove punctuations and special characters
    # string = re.sub(r'[^\w\s]','', string) # remove tanda baca
    string = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    # Menghapus enter
    string = re.sub(r"\n", "", string)

    # Membersihkan elemen yang tidak perlu, seperti menghapus spasi 2
    string = re.sub(r"\'re", " \'re", string)

    # Mengecek digit atau bukan
    string = re.sub(r"\'d", " \'d", string)

    # Mengecek long atau bukan
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip()
    # Menghilangkan imbuhan
    
    return string

def text_normalize(string):
    words = string.split()
    normalized_text = ' '.join([key_norm[key_norm['slang'] == word]['formal'].values[0] if (key_norm['slang'] == word).any() else word for word in words])
    return normalized_text.lower()

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\bsampah\b', 'sampahnya', text)
    text = casefolding(text)
    text = text_normalize(text)
    return text

@app.get("/", description="This is our first route.")
async def root():
    return {"message": "Hello we are from Group 2"}

@app.post("/api/sentiment", description="This is sentiment analysis route")
async def post(feedback: Feedback):
    text = feedback.text

    text_processed = preprocess_text(text)

    # Load tokenizer's configuration
    with open('./models/bilstm_fix/tokenizer_config_bilstm_fix3.pkl', 'rb') as f:
        tokenizer_config = pickle.load(f)

    # Recreate tokenizer
    tokenizer = Tokenizer(**tokenizer_config)

    # Load tokenizer's word index
    with open('./models/bilstm_fix/tokenizer_word_index_bilstm_fix3.pkl', 'rb') as f:
        tokenizer.word_index = pickle.load(f)

    # new data to predict
    new_texts = [text_processed]

    # Tokenize new data
    sequences = tokenizer.texts_to_sequences(new_texts)

    # Padding sequences
    max_length = 100  # model's max_length
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    # Load the saved model
    model = load_model('./models/bilstm_fix/best_model3.keras')

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

if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)