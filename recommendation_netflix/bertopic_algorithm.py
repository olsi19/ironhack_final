import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import random

# Load the Netflix dataset
netflix_dataset = pd.read_csv('./netflix_dataset.csv')

# Filter out movies and TV shows into separate DataFrames
movies = netflix_dataset[netflix_dataset['type'] == 'Movie'].copy()
tv_shows = netflix_dataset[netflix_dataset['type'] == 'TV Show'].copy()

# Fill missing descriptions with empty strings
movies['description'] = movies['description'].fillna('')

# Initialize BERTopic model
topic_model = BERTopic()

# Fit the model to get topics (not used further in this code)
topics, _ = topic_model.fit_transform(movies['description'])

# Get the underlying SentenceTransformer model
sentence_model = topic_model.embedding_model.embedding_model

# Generate embeddings for movie descriptions
embeddings = sentence_model.encode(movies['description'].tolist(), show_progress_bar=True)

# Calculate the cosine similarity matrix for the embeddings
cos_sim = cosine_similarity(embeddings)

# Convert movie titles to lowercase and create a Series to map movie titles to their indices
movies['title_lower'] = movies['title'].str.lower()
indexes = pd.Series(movies.index, index=movies['title_lower']).drop_duplicates()

# Convert the cosine similarity matrix into a DataFrame and set the column names to the lowercase titles
cos_sim_df = pd.DataFrame(cos_sim, index=indexes.index, columns=indexes.index)


# Function to get movie recommendations using BERTopic
def get_bertopic_recommendations(title):
    title_lower = title.lower()
    if title_lower not in indexes.index:
        return None

    # Get the similarity scores for the given title
    sim_scores = cos_sim_df[title_lower].sort_values(ascending=False)

    # Exclude the input title from recommendations and select top 10 unique titles
    recommendations = movies.loc[sim_scores.index != title_lower, ['title', 'description']].drop_duplicates(
        subset='title').head(10)

    return recommendations


# Streamlit UI
st.title("NETFLIX RECOMMENDATION SYSTEM")

st.subheader("Suggested Titles:")
random_titles = random.sample(list(movies['title']), 5)
for title in random_titles:
    st.text(title)

title = st.text_input("Enter a movie title:")
if st.button("Show Recommendations"):
    if title:
        recommendations = get_bertopic_recommendations(title)
        if recommendations is not None and not recommendations.empty:
            st.subheader("Top 10 Recommendations:")
            for idx, row in recommendations.iterrows():
                st.text(f"Title: {row['title']}")
                st.text(f"Description: {row['description']}")
                st.text("")
        else:
            st.error(f"No recommendations found for title '{title}'. Please try a different title.")
    else:
        st.error("Please enter a movie title.")
