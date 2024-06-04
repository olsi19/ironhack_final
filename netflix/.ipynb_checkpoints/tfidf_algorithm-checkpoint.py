import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load the Netflix dataset
netflix_dataset = pd.read_csv('./netflix_dataset.csv')

# Filter out movies and TV shows into separate DataFrames
movies = netflix_dataset[netflix_dataset['type'] == 'Movie'].copy()
tv_shows = netflix_dataset[netflix_dataset['type'] == 'TV Show'].copy()

# Fill missing descriptions with empty strings
movies['description'] = movies['description'].fillna('')

# Initialize a TF-IDF Vectorizer to convert text to numerical data
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the descriptions of movies into a TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movies['description'])

# Calculate the cosine similarity matrix for the TF-IDF matrix
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Convert movie titles to lowercase and create a Series to map movie titles to their indices
movies['title_lower'] = movies['title'].str.lower()
indexes = pd.Series(movies.index, index=movies['title_lower']).drop_duplicates()

# Convert the cosine similarity matrix into a DataFrame and set the column names to the lowercase titles
cos_sim_df = pd.DataFrame(cos_sim)
cos_sim_df.columns = indexes.index
cos_sim_df['title'] = indexes.index
cos_sim_df = cos_sim_df.set_index('title')


# Function to get movie recommendations
def show_recommendation(title, cos_sim=cos_sim_df):
    title_lower = title.lower()
    if title_lower not in indexes.index:
        return None
    idx = indexes[title_lower]
    sim_scores = list(enumerate(cos_sim_df.loc[title_lower]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:]  # skip the first one as it is the same movie

    # Get unique recommendations
    seen_titles = set()
    recommendations = []
    for i in sim_scores:
        movie_idx = i[0]
        movie_title = movies.iloc[movie_idx]['title']
        if movie_title not in seen_titles:
            seen_titles.add(movie_title)
            recommendations.append(movie_idx)
        if len(recommendations) == 10:
            break

    return movies.iloc[recommendations][['title', 'description']]


# Streamlit UI
st.title("NETFLIX RECOMMENDATION SYSTEM")

st.subheader("Suggested Titles:")
random_titles = random.sample(list(movies['title']), 5)
for title in random_titles:
    st.text(title)

title = st.text_input("Enter a movie title:")
if st.button("Show Recommendations"):
    if title:
        recommendations = show_recommendation(title)
        if recommendations is not None:
            st.subheader("Top 10 Recommendations:")
            for idx, row in recommendations.iterrows():
                st.text(f"Title: {row['title']}")
                st.text(f"Description: {row['description']}")
                st.text("")
        else:
            st.error(f"Title '{title}' not found in the dataset.")
    else:
        st.error("Please enter a movie title.")
