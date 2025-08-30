import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# تحميل الداتا
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    return movies

movies = load_data()

# تجهيز TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(''))

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# دالة الترشيح
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'].str.contains(title, case=False, na=False)].index
    if len(idx) == 0:
        return []
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# واجهة Streamlit
st.title("🎬 Movie Recommender")
movie_name = st.text_input("اكتب اسم فيلم:")

if movie_name:
    results = get_recommendations(movie_name)
    if len(results) == 0:
        st.write("❌ الفيلم مش موجود في الداتا.")
    else:
        st.write("✅ أفلام مقترحة:")
        for r in results:
            st.write("- ", r)
