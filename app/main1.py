from fastapi import FastAPI
from recommender import recommend_songs

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Song Recommendation System API"}

@app.get("/recommend/")
def recommend(song: str):
    return {"recommended_songs": recommend_songs(song)}
