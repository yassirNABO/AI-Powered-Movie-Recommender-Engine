import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Scalability: Simulating a dataset of 5,000 movies
print("Initializing Movie Engine with 5,000 titles...")
movies = [f"Movie_{i}" for i in range(5000)]
genres = ['Action', 'Sci-Fi', 'Drama', 'Comedy', 'Horror', 'Romance']

# Generate random metadata for each movie
data = {
    'Movie_Title': movies,
    'Description': [f"A thrilling {np.random.choice(genres)} story about {np.random.choice(['space', 'crime', 'love', 'future', 'war'])}." for _ in range(5000)],
    'Genre': [np.random.choice(genres) for _ in range(5000)]
}

df = pd.DataFrame(data)

# 2. NLP Component: TF-IDF Vectorization
# This converts text (descriptions) into a numerical matrix that the AI can understand
print("Vectorizing descriptions using TF-IDF...")
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Description'])

# 3. Computing Similarity Matrix (The Big Data "Brain")
# This calculates how similar every movie is to every other movie (5000x5000 matrix)
print("Computing Cosine Similarity Matrix...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 4. Recommendation Function
def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        idx = df.index[df['Movie_Title'] == title][0]
        # Get similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Sort them based on similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Get top 5 most similar movies
        sim_scores = sim_scores[1:6]
        movie_indices = [i[0] for i in sim_scores]
        return df['Movie_Title'].iloc[movie_indices]
    except:
        return "Movie not found."

# Test the Engine
test_movie = "Movie_42"
print(f"\nTop 5 Recommendations for '{test_movie}':")
print(get_recommendations(test_movie))
