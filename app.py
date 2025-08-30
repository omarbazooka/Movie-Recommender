import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# تحميل البيانات
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
        return pd.DataFrame()
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres']]

# واجهة Streamlit
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# CSS للتجميل
st.markdown("""
    <style>
    .title {
        font-size: 40px;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
    }
    .movie-card {
        background: #1E1E1E;
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🎬 Movie Recommender System</p>', unsafe_allow_html=True)

# إدخال الفيلم
movie_name = st.text_input("اكتب اسم فيلم:", placeholder="مثال: Toy Story")

if movie_name:
    results = get_recommendations(movie_name)
    if results.empty:
        st.warning("❌ الفيلم مش موجود في الداتا.")
    else:
        st.success("✅ أفلام مقترحة لك:")
        cols = st.columns(3)
        for i, row in enumerate(results.itertuples(), 1):
            with cols[(i-1) % 3]:
                st.markdown(f"""
                    <div class="movie-card">
                        <h4>{row.title}</h4>
                        <p>{row.genres}</p>
                    </div>
                """, unsafe_allow_html=True)
