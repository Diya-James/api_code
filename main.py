from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from fastapi import FastAPI
import json







# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# model = hub.load(module_url)

app = FastAPI()

def clean_offence(offence):
  section =  re.sub("[^a-zA-Z0-9 ]", "", offence)

  return offence

@app.get("/title/{title_id}")
def read_root(title_id: str):
    ipc = pd.read_csv("ipc_data.csv")
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    ipc["clean_offence"] = ipc["Offence"].apply(clean_offence)
    tfidf = vectorizer.fit_transform(ipc["clean_offence"])
    query_vec = vectorizer.transform([title_id])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -3)[-3:]
    results = ipc.iloc[indices].iloc[::-1].to_json(orient="records")
    parsed = json.loads(results)
    return parsed

# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}



# To Run
# uvicorn main:app --reload

# Eg: http://127.0.0.1:8000/title/murder