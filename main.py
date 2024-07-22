import pickle

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import Union

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import re
import pandas as pd

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from mpstemmer import MPStemmer

factory = StemmerFactory()
stemmer = factory.create_stemmer()
lemma = MPStemmer()

key_norm = pd.read_csv('../corpus/colloquial-indonesian-lexicon.csv')

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

# Stopword remover setup
stop_words = StopWordRemoverFactory().get_stop_words()
new_array = ArrayDictionary(stop_words)
stop_words_remover = StopWordRemover(new_array)

def remove_stopwords(text):
    return stop_words_remover.remove(text)

def stemming(text):
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def lemmatization(text):
    return lemma.stem(text)

# Function to preprocess text
def preprocess_text(feedback):
    text = casefolding(feedback)
    text = text_normalize(text)
    text = remove_stopwords(text)
    text = stemming(text)
    text = lemmatization(text)
    return text

app = FastAPI()

class Feedback(BaseModel):
    email: Union[str, None] = None
    text: str

@app.get("/", description="This is our first route.")
async def root():
    return {"message": "Hello we are from Group 2"}

@app.post("/api/sentiment", description="This is sentiment analysis route")
async def post(feedback: Feedback):
    text = feedback.text

    text_processed = preprocess_text(text)

    # Load tokenizer's configuration
    with open('./models/asli_fix/tokenizer_config_w2v_bilstm_fix.pkl', 'rb') as f:
        tokenizer_config = pickle.load(f)

    # Recreate tokenizer
    tokenizer = Tokenizer(**tokenizer_config)

    # Load tokenizer's word index
    with open('./models/asli_fix/tokenizer_word_index_w2v_bilstm_fix.pkl', 'rb') as f:
        tokenizer.word_index = pickle.load(f)

    # new data to predict
    new_texts = [text_processed]

    # Tokenize new data
    sequences = tokenizer.texts_to_sequences(new_texts)

    # Padding sequences
    max_length = 100  # model's max_length
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    # Load the saved model
    model = load_model('./models/asli_fix/model_w2v_bilstm_fix.keras')

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
  download_nltk_data()
  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)