import pandas as pd
import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load the Netflix dataset
netflix_dataset = pd.read_csv('./netflix_dataset.csv')

# Fill missing descriptions with empty strings
netflix_dataset['description'] = netflix_dataset['description'].fillna('')

# Initialize a TF-IDF Vectorizer to convert text to numerical data
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the descriptions of movies into a TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(netflix_dataset['description'])

# Calculate the cosine similarity matrix for the TF-IDF matrix
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Convert movie titles to lowercase and create a Series to map movie titles to their indices
netflix_dataset['title_lower'] = netflix_dataset['title'].str.lower()
indexes = pd.Series(netflix_dataset.index,
                    index=netflix_dataset['title_lower']).drop_duplicates()

# Function to fetch movie info from OMDB API


def fetch_movie_info(title):
    api_key = '3edc7868'  # OMDB API key
    url = f'http://www.omdbapi.com/?t={title}&apikey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Function to get movie recommendations


def show_recommendation(title, diversity_factor=20):
    movie_info = fetch_movie_info(title)
    if movie_info and movie_info['Response'] == 'True':
        description = movie_info.get('Plot', '')
        poster = movie_info.get('Poster', '')

        if description:
            # Add the fetched description to the dataframe temporarily
            temp_row = pd.DataFrame(
                {'title': [title], 'description': [description]})
            temp_df = pd.concat([netflix_dataset, temp_row], ignore_index=True)
            temp_df['description'] = temp_df['description'].fillna('')

            # Recompute the TF-IDF matrix and cosine similarity with the new description
            tfidf_matrix_temp = tfidf.fit_transform(temp_df['description'])
            cos_sim_temp = cosine_similarity(
                tfidf_matrix_temp, tfidf_matrix_temp)
            cos_sim_temp_df = pd.DataFrame(cos_sim_temp)
            cos_sim_temp_df.columns = temp_df.index
            cos_sim_temp_df['title'] = temp_df['title']
            cos_sim_temp_df = cos_sim_temp_df.set_index('title')

            title_lower = title.lower()
            idx = temp_df[temp_df['title'].str.lower() == title_lower].index[0]
            sim_scores = list(enumerate(cos_sim_temp_df.loc[title]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # skip the first one as it is the same movie
            sim_scores = sim_scores[1:]

            # Randomize within the top N results for diversity
            top_indices = [i for i, _ in sim_scores[:diversity_factor]]
            random.shuffle(top_indices)

            # Keep track of recommended titles
            recommended_titles = set()
            recommendations = []

            # Iterate over top indices and add unique recommendations
            for i in top_indices:
                movie_title = netflix_dataset.iloc[i]['title']
                if movie_title not in recommended_titles:
                    recommended_titles.add(movie_title)
                    recommendations.append(i)
                    if len(recommendations) == 5:
                        break

            # Retrieve posters for the recommended movies
            recommended_movies = netflix_dataset.iloc[recommendations][[
                'title', 'description']]
            recommended_movies['poster'] = recommended_movies['title'].apply(
                lambda x: fetch_movie_info(x).get('Poster', ''))

            return recommended_movies, poster
        else:
            return None, None
    else:
        return None, None


# Streamlit UI
st.set_page_config(
    page_title="MOVIES RECOMMENDATION SYSTEM",
    page_icon=":clapper:",
    layout="centered",  # Wide layout to make room for recommendations
    initial_sidebar_state="collapsed"  # Collapsed sidebar by default
)
# Add Netflix logo
st.image(r"C:\Users\zaiid\Downloads\580b57fcd9996e24bc43c529 (2).png",
         use_column_width=False, width=300)
st.title("MOVIES RECOMMENDATION SYSTEM")
title = st.text_input("Title:")

if title:
    movie_info = fetch_movie_info(title)
    if movie_info and movie_info['Response'] == 'True':
        poster = movie_info.get('Poster', '')
        if poster:
            st.image(poster, caption=title, use_column_width=False, width=200)

if st.button("Show Recommendations"):
    if title:
        recommendations, poster = show_recommendation(title)
        if recommendations is not None:
            st.subheader("Top 5 Recommendations:")
            for idx, row in recommendations.iterrows():
                try:
                    st.write("")  # Add an empty line for spacing
                    col1, col2 = st.columns([2, 5])
                    with col1:
                        st.image(row['poster'], caption=row['title'],
                                 use_column_width=False, width=150)
                    with col2:
                        st.write(f"<div style='margin-top: 20px;'>"
                                 f"<b>Title:</b> {row['title']}<br>"
                                 f"<b>Description:</b> {row['description']}</div>",
                                 unsafe_allow_html=True)
                    st.write("")  # Add an empty line for spacing
                except:
                    continue
        else:
            st.error("No recommendations found.")
    else:
        st.error("Please enter a movie title.")
