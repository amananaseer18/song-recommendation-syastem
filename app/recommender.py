import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/songs.csv")

df['combined_features'] = (
    df['title'] + " " +
    df['artist'] + " " +
    df['genre'] + " " +
    df['lyrics']
)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_songs(song_title, num_recommendations=5):
    if song_title not in df['title'].values:
        return ["Song not found"]

    index = df[df['title'] == song_title].index[0]
    similarity_scores = list(enumerate(cosine_sim[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended_indices = [i[0] for i in similarity_scores[1:num_recommendations+1]]
    return df['title'].iloc[recommended_indices].tolist()
